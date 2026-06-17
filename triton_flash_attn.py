# PyTorch は tensor の確保、autograd.Function の定義、参照実装との比較に使う。
import torch

# triton は kernel の JIT compile、autotune、launch grid、benchmark 補助などを提供する。
import triton
# tl は Triton kernel 内で使う DSL。tl.load, tl.store, tl.dot, tl.arange などを含む。
import triton.language as tl


# @triton.jit は、この Python 関数を GPU kernel 用の Triton IR に JIT compile する指定。
# この関数は単独 kernel として launch されるのではなく、_attn_fwd の内部で inline 的に使われる補助関数。
@triton.jit
def _attn_fwd_inner(
    # O_block は現在の query block に対する出力 accumulator。形状は [BLOCK_SIZE_Q, HEAD_DIM]。
    O_block,
    # l_i は online softmax の分母の running sum。query 行ごとに 1 つ持つ。
    l_i,
    # m_i は online softmax の running max。数値安定化のため query 行ごとに 1 つ持つ。
    m_i,
    # Q_block はこの Triton program が担当する query tile。通常 SRAM/register に保持される。
    Q_block,
    # K_block_ptr は K の block pointer。tl.load(K_block_ptr) で K tile を読む。
    K_block_ptr,
    # V_block_ptr は V の block pointer。tl.load(V_block_ptr) で V tile を読む。
    V_block_ptr,
    # block_index_q は sequence 軸の query block ID。tl.program_id(0) から来る。
    block_index_q,
    # softmax_scale は通常 1 / sqrt(HEAD_DIM)。attention score のスケール係数。
    softmax_scale,
    # BLOCK_SIZE_Q は query 方向の tile size。tl.constexpr は compile-time constant を表す。
    BLOCK_SIZE_Q: tl.constexpr,
    # BLOCK_SIZE_KV は key/value 方向の tile size。loop の刻み幅にもなる。
    BLOCK_SIZE_KV: tl.constexpr,
    # STAGE は causal / non-causal のどの範囲を処理するかを compile 時に分岐させる定数。
    STAGE: tl.constexpr,
    # offs_q は query token の offset vector。mask 作成に使う。
    offs_q: tl.constexpr,
    # offs_kv は key/value token の offset vector。mask 作成に使う。
    offs_kv: tl.constexpr,
    # SEQ_LEN は sequence length。block pointer の boundary や loop 終端に使う。
    SEQ_LEN: tl.constexpr,
):
    # STAGE ごとに、この inner loop がどの key/value 範囲を処理するかを決める。
    # causal attention では、query block より左側は全て attend 可能、対角 block は mask が必要。

    # STAGE == 1 は「causal の対角より左側」だけを処理するケース。
    if STAGE == 1:
        # lo は K/V の開始位置、hi は query block の左端まで。ここは causal mask 不要。
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    # STAGE == 2 は causal の対角を含む block。token 単位の mask が必要。
    elif STAGE == 2:
        # この query block と同じ位置の K/V block だけを処理する。
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        # tl.multiple_of は compiler hint。「lo は BLOCK_SIZE_Q の倍数」と伝えて最適化を助ける。
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    # それ以外は non-causal attention。全 key/value を処理する。
    else:
        # non-causal では全 query が全 key/value に attend できるので範囲は [0, SEQ_LEN)。
        lo, hi = 0, SEQ_LEN

    # K_block_ptr を key/value loop の開始位置 lo まで進める。
    # K は block pointer 上では shape=(HEAD_DIM, SEQ_LEN) として扱っているため、列方向に lo 進める。
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # V_block_ptr を value sequence の開始位置 lo まで進める。
    # V は shape=(SEQ_LEN, HEAD_DIM) として扱っているため、行方向に lo 進める。
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # K/V を BLOCK_SIZE_KV ずつ読み、online softmax の統計量と出力 accumulator を更新する。
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # start_kv が BLOCK_SIZE_KV の倍数であることを compiler に伝える。
        # これにより alignment や loop 最適化が入りやすくなる。
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # tl.load は GPU memory から tile を読む。block pointer なので block_shape 分をまとめて読む。
        K_block = tl.load(K_block_ptr)
        # tl.dot は tile 行列積。ここでは Q_block [BQ, HD] x K_block [HD, BKV] -> QK_block [BQ, BKV]。
        # dtype と shape が条件を満たすと Tensor Core / MFMA などに lower される可能性がある。
        QK_block = tl.dot(Q_block, K_block)

        # STAGE == 2 は causal の対角 block なので、query index < key index の要素を mask する。
        if STAGE == 2:
            # mask は [BQ, BKV]。True なら attend 可能、False なら未来 token なので mask する。
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            # score に softmax_scale を掛け、mask される要素には大きな負値を足す。
            # -inf の代わりに -1e6 を使い、exp 後にほぼ 0 になるようにする。
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            # m_ij は新しい block を見た後の row-wise max。
            # tl.max(QK_block, 1) は各 query 行に対して key/value 方向の最大値を取る。
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            # softmax の数値安定化のため、新しい max を各行から引く。
            QK_block -= m_ij[:, None]
        # 対角 block 以外、または non-causal の場合は mask なし。
        else:
            # score の max を取ってから softmax_scale を掛けている点に注意。
            # 数式上は max(score * scale) と scale * max(score) は scale > 0 なら同値。
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            # score に scale を掛け、新しい row-wise max を引いて安定化する。
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # exp(score - max) を計算する。ここで P_block はまだ正規化前の softmax numerator。
        P_block = tl.math.exp(QK_block)
        # query 行ごとに numerator を合計し、新しい block が寄与する softmax 分母を得る。
        l_ij = tl.sum(P_block, 1)

        # alpha は古い max m_i で管理していた分母・accumulator を、新しい max m_ij の基準へ変換する係数。
        # exp(m_old - m_new) なので、m_new >= m_old のとき 0〜1 の範囲になる。
        alpha = tl.math.exp(m_i - m_ij)
        # 古い分母 l_i を alpha で再スケールし、新しい block の分母 l_ij を足す。
        l_i = l_i * alpha + l_ij

        # V tile を読む。形状は [BLOCK_SIZE_KV, HEAD_DIM]。
        V_block = tl.load(V_block_ptr)
        # P_block を float16 に落として tl.dot(P, V) の Tensor Core 利用を狙う。
        # 精度を重視する場合は dtype 設計を別途検討する。
        P_block = P_block.to(tl.float16)
        # 出力 accumulator も古い max 基準から新しい max 基準へ alpha で再スケールする。
        O_block = O_block * alpha[:, None]
        # tl.dot(P_block, V_block, O_block) は P @ V を計算し、既存 accumulator O_block に加算する形。
        # つまり O_block = O_block + P_block @ V_block。
        O_block = tl.dot(P_block, V_block, O_block)

        # running max を更新する。
        m_i = m_ij

        # 次の V block へ進める。V は sequence 方向が行なので (BLOCK_SIZE_KV, 0)。
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        # 次の K block へ進める。K は転置 view として扱っているので列方向が sequence。
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    # 更新後の accumulator、softmax 分母、logsumexp 用 max を返す。
    return O_block, l_i, m_i


# @triton.autotune は複数の Config を試し、key で指定した problem size ごとに高速な設定を選ぶ。
@triton.autotune(
    # Config のリスト。BLOCK_SIZE_Q/KV、num_stages、num_warps の候補を全列挙する。
    [
        # triton.Config は meta-parameter と launch 設定をまとめるオブジェクト。
        triton.Config(
            # meta-parameter。kernel 引数の tl.constexpr として compile-time に使われる。
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            # num_stages は software pipelining の段数。load と compute の overlap に関係する。
            num_stages=num_stages,
            # num_warps は 1 Triton program instance を実行する warp 数。
            num_warps=num_warps,
        )
        # query block size の候補。
        for BLOCK_SIZE_Q in [64, 128]
        # key/value block size の候補。
        for BLOCK_SIZE_KV in [32, 64]
        # pipeline stage 数の候補。
        for num_stages in ([3, 4, 7])
        # program あたり warp 数の候補。
        for num_warps in [2, 4]
    ],
    # SEQ_LEN と HEAD_DIM が同じ問題では autotune 結果を再利用する。
    key=["SEQ_LEN", "HEAD_DIM"],
)
# _attn_fwd は forward attention 本体の Triton kernel。
@triton.jit
def _attn_fwd(
    # Q pointer。論理 shape は [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]。
    Q,
    # K pointer。論理 shape は [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]。
    K,
    # V pointer。論理 shape は [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]。
    V,
    # softmax_scale は attention score に掛ける係数。通常 1 / sqrt(HEAD_DIM)。
    softmax_scale,
    # M は backward 用に保存する logsumexp。shape は [BATCH_SIZE, NUM_HEADS, SEQ_LEN]。
    M,
    # O pointer。出力 tensor。shape は Q と同じ。
    O,
    # Q の batch stride。PyTorch の stride は byte ではなく element 単位。
    stride_Q_batch,
    # Q の head stride。
    stride_Q_head,
    # Q の sequence stride。
    stride_Q_seq,
    # Q の head_dim stride。
    stride_Q_dim,
    # K の batch stride。
    stride_K_batch,
    # K の head stride。
    stride_K_head,
    # K の sequence stride。
    stride_K_seq,
    # K の head_dim stride。
    stride_K_dim,
    # V の batch stride。
    stride_V_batch,
    # V の head stride。
    stride_V_head,
    # V の sequence stride。
    stride_V_seq,
    # V の head_dim stride。
    stride_V_dim,
    # O の batch stride。
    stride_O_batch,
    # O の head stride。
    stride_O_head,
    # O の sequence stride。
    stride_O_seq,
    # O の head_dim stride。
    stride_O_dim,
    # BATCH_SIZE は runtime 引数だが、この kernel 内では主に host 側 grid 計算で使われる。
    BATCH_SIZE,
    # NUM_HEADS は compile-time constant。batch-head index の復元に使う。
    NUM_HEADS: tl.constexpr,
    # SEQ_LEN は compile-time constant。block pointer shape と loop bounds に使う。
    SEQ_LEN: tl.constexpr,
    # HEAD_DIM は compile-time constant。tile の列数と tl.dot の K 次元。
    HEAD_DIM: tl.constexpr,
    # BLOCK_SIZE_Q は autotune で選ばれる query tile size。
    BLOCK_SIZE_Q: tl.constexpr,
    # BLOCK_SIZE_KV は autotune で選ばれる key/value tile size。
    BLOCK_SIZE_KV: tl.constexpr,
    # STAGE は causal なら 3、non-causal なら 1 として渡される。
    STAGE: tl.constexpr,
):
    # tl.static_assert は compile-time assertion。条件を満たさない config を compile 時に弾く。
    # このコードでは BLOCK_SIZE_KV <= HEAD_DIM を仮定している。
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # tl.program_id(0) は launch grid の第 0 軸の program ID。
    # ここでは sequence 上の query block index を表す。
    block_index_q = tl.program_id(0)

    # tl.program_id(1) は launch grid の第 1 軸の program ID。
    # ここでは batch と head を flatten した index を表す。
    index_batch_head = tl.program_id(1)
    # flatten された batch-head index から batch index を復元する。
    index_batch = index_batch_head // NUM_HEADS
    # flatten された batch-head index から head index を復元する。
    index_head = index_batch_head % NUM_HEADS

    # Q/K/V/O のうち、このコードでは Q の stride を使って batch/head の base offset を計算している。
    # 注意: K/V/O が Q と異なる layout の場合は、それぞれの stride で base offset を計算すべき。
    qvk_offset = (
        # int64 に cast して大きな tensor でも pointer offset overflow を避ける。
        index_batch.to(tl.int64) * stride_Q_batch
        # head 方向の offset を足す。
        + index_head.to(tl.int64) * stride_Q_head
    )

    # tl.make_block_ptr は base pointer、親 tensor shape、strides、offsets、block_shape から block pointer を作る。
    Q_block_ptr = tl.make_block_ptr(
        # Q の batch/head slice の base address。
        base=Q + qvk_offset,
        # この block pointer から見た 2D 親 tensor shape は [SEQ_LEN, HEAD_DIM]。
        shape=(SEQ_LEN, HEAD_DIM),
        # 2D 親 tensor の stride。PyTorch と同じく element 単位。
        strides=(stride_Q_seq, stride_Q_dim),
        # この program が読む query block の開始位置。
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        # 読む block の形状。
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        # memory order hint。最後の次元 HEAD_DIM を連続方向として扱う意図。
        order=(1, 0),
    )

    # V の block pointer。V は [SEQ_LEN, HEAD_DIM] として読み、KV loop 内で行方向に進める。
    V_block_ptr = tl.make_block_ptr(
        # 注意: ここも qvk_offset を使うため、V の batch/head stride が Q と同じである前提。
        base=V + qvk_offset,
        # V の 2D view shape。
        shape=(SEQ_LEN, HEAD_DIM),
        # V の sequence/head_dim stride。
        strides=(stride_V_seq, stride_V_dim),
        # 最初は sequence 0 から読む。inner で lo まで tl.advance する。
        offsets=(0, 0),
        # 一度に読む V block の形状は [BLOCK_SIZE_KV, HEAD_DIM]。
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        # HEAD_DIM 方向を連続方向とする order hint。
        order=(1, 0),
    )

    # K の block pointer。QK = Q @ K^T を行うため、K を転置 view [HEAD_DIM, SEQ_LEN] として読む。
    K_block_ptr = tl.make_block_ptr(
        # 注意: ここも qvk_offset を使うため、K の batch/head stride が Q と同じである前提。
        base=K + qvk_offset,
        # K を転置して見た shape は [HEAD_DIM, SEQ_LEN]。
        shape=(HEAD_DIM, SEQ_LEN),
        # 転置 view なので stride の順序も [dim stride, seq stride] にする。
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),
        # 最初は dim=0, seq=0 から読む。inner で key sequence 側に進める。
        offsets=(0, 0),
        # 読む K block は [HEAD_DIM, BLOCK_SIZE_KV]。
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        # K block の order hint。shape と access pattern に合わせて compiler に layout 意図を伝える。
        order=(0, 1),
    )

    # O の block pointer。出力 O の query block に store するために使う。
    O_block_ptr = tl.make_block_ptr(
        # 注意: O の batch/head stride が Q と異なる場合は O 用 offset を別に計算すべき。
        base=O + qvk_offset,
        # O の 2D view shape。
        shape=(SEQ_LEN, HEAD_DIM),
        # O の sequence/head_dim stride。
        strides=(stride_O_seq, stride_O_dim),
        # この program が書き込む query block の開始位置。
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        # 書き込む block の形状。
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        # HEAD_DIM 方向を連続方向とする order hint。
        order=(1, 0),
    )

    # offs_q はこの query block 内の絶対 token index。shape は [BLOCK_SIZE_Q]。
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv は key/value block 内の相対 token index。shape は [BLOCK_SIZE_KV]。
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i は online softmax の running maximum。初期値は -inf。
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i は online softmax の running denominator。初期値を 1.0 にしている。
    # 一般的な説明では 0 から始めるが、m_i=-inf なので最初の alpha=0 になり、実質的に上書きされる。
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # O_block は出力 accumulator。P@V の累積を float32 で保持する。
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Q_block を一度だけ読む。FlashAttention では Q tile を保持したまま K/V tile を走査する。
    Q_block = tl.load(Q_block_ptr)

    # STAGE は host 側で causal=True なら 3、causal=False なら 1 に設定される。
    # この実装では inner に渡す STAGE を 4 - STAGE と 2 に分け、causal/non-causal を処理する。

    # non-causal の全範囲、または causal の左側 block を処理する分岐。
    if STAGE == 1 or STAGE == 3:
        # _attn_fwd_inner は K/V block を sweep し、O_block, l_i, m_i を更新する。
        O_block, l_i, m_i = _attn_fwd_inner(
            # 現在の出力 accumulator。
            O_block,
            # 現在の softmax denominator。
            l_i,
            # 現在の softmax max。
            m_i,
            # 固定して使う Q tile。
            Q_block,
            # K block pointer。
            K_block_ptr,
            # V block pointer。
            V_block_ptr,
            # query block index。
            block_index_q,
            # score scale。
            softmax_scale,
            # query block size。
            BLOCK_SIZE_Q,
            # key/value block size。
            BLOCK_SIZE_KV,
            # causal=True のとき 4-3=1、non-causal のとき 4-1=3。
            # inner 側では 1 が左側 causal、3 が non-causal として扱われる。
            4 - STAGE,
            # query offset vector。
            offs_q,
            # key/value offset vector。
            offs_kv,
            # sequence length。
            SEQ_LEN,
        )

    # causal attention の場合だけ、対角 block を mask 付きで処理する。
    if STAGE == 3:
        # STAGE=2 を inner に渡すと causal mask を使う対角 block 処理になる。
        O_block, l_i, m_i = _attn_fwd_inner(
            # 現在の出力 accumulator。
            O_block,
            # 現在の softmax denominator。
            l_i,
            # 現在の softmax max。
            m_i,
            # 固定して使う Q tile。
            Q_block,
            # K block pointer。inner 内で lo まで advance する。
            K_block_ptr,
            # V block pointer。inner 内で lo まで advance する。
            V_block_ptr,
            # query block index。
            block_index_q,
            # score scale。
            softmax_scale,
            # query block size。
            BLOCK_SIZE_Q,
            # key/value block size。
            BLOCK_SIZE_KV,
            # causal diagonal stage。
            2,
            # query offset vector。
            offs_q,
            # key/value offset vector。
            offs_kv,
            # sequence length。
            SEQ_LEN,
        )
    # epilogue は loop 後の正規化と backward 用 metadata 保存。
    # m_i はここで logsumexp = m_i + log(l_i) に変換される。
    m_i += tl.math.log(
        # l_i は exp(score - m_i) の行方向合計。
        l_i
    )
    # accumulator はまだ numerator の weighted sum なので、分母 l_i で割って softmax 正規化する。
    O_block = O_block / l_i[:, None]
    # M の保存先 pointer。M は contiguous [B*H, SEQ_LEN] として扱われている。
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    # backward で使う logsumexp を store する。
    tl.store(m_ptrs, m_i)
    # O_block を O tensor の dtype に cast して保存する。
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


# backward の前処理 kernel。D_i = sum_j O_ij * dO_ij を各 query 行について計算する。
@triton.jit
def _attn_bwd_preprocess(
    # forward 出力 O pointer。
    O,
    # upstream gradient dO pointer。
    dO,
    # D pointer。softmax backward で使う row-wise dot(O, dO) を保存する。
    D,
    # sequence length。
    SEQ_LEN,
    # query block size。
    BLOCK_SIZE_Q: tl.constexpr,
    # head dimension。
    HEAD_DIM: tl.constexpr,
):
    # sequence 軸の query block index。
    block_index_q = tl.program_id(0)
    # この block が担当する query token の絶対 offset。
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # batch-head を flatten した index。
    index_batch_head = tl.program_id(1)
    # head_dim 方向の offset。
    offs_dim = tl.arange(0, HEAD_DIM)
    # O block を load する。ここは stride を使わず contiguous [B*H, SEQ_LEN, HEAD_DIM] 前提。
    O_block = tl.load(
        # base pointer。
        O
        # batch-head slice の offset。
        + index_batch_head * HEAD_DIM * SEQ_LEN
        # sequence row offset。
        + offs_q[:, None] * HEAD_DIM
        # head_dim column offset。
        + offs_dim[None, :]
    )
    # dO block を load する。形状は O_block と同じ [BLOCK_SIZE_Q, HEAD_DIM]。
    dO_block = tl.load(
        # base pointer。
        dO
        # batch-head slice の offset。
        + index_batch_head * HEAD_DIM * SEQ_LEN
        # sequence row offset。
        + offs_q[:, None] * HEAD_DIM
        # head_dim column offset。
        + offs_dim[None, :]
    ).to(tl.float32)
    # D_i = sum_d dO_i,d * O_i,d。softmax backward の Delta に相当する。
    D_block = tl.sum(dO_block * O_block, axis=1)
    # D の保存先 pointer。D は [B*H, SEQ_LEN] contiguous として扱う。
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    # D_block を global memory に store する。
    tl.store(D_block_ptrs, D_block)


# dQ を計算する backward kernel。
# Q block を固定し、全 K/V block を走査して dQ を累積する。
@triton.jit
def _attn_bwd_dq(
    # Q pointer。
    Q,
    # K pointer。
    K,
    # V pointer。
    V,
    # softmax scale。
    softmax_scale,
    # upstream gradient dO pointer。
    dO,
    # dQ output pointer。
    dQ,
    # dK output pointer。この kernel では実質使わないが signature を揃えている。
    dK,
    # dV output pointer。この kernel では実質使わないが signature を揃えている。
    dV,
    # forward で保存した logsumexp M pointer。
    M,
    # backward 前処理で計算した D pointer。
    D,
    # tensor の batch stride。
    stride_batch,
    # tensor の head stride。
    stride_head,
    # tensor の sequence stride。
    stride_seq,
    # tensor の head_dim stride。
    stride_dim,
    # number of heads。
    NUM_HEADS,
    # sequence length。
    SEQ_LEN,
    # query block size。
    BLOCK_Q: tl.constexpr,
    # key/value block size。
    BLOCK_KV: tl.constexpr,
    # head dimension。
    HEAD_DIM: tl.constexpr,
    # causal/non-causal stage。
    STAGE: tl.constexpr,
):
    # grid 第 2 軸は batch-head index。
    index_batch_head = tl.program_id(2)
    # flatten index から batch index を復元する。
    index_batch = index_batch_head // NUM_HEADS
    # flatten index から head index を復元する。
    index_head = index_batch_head % NUM_HEADS
    # batch/head slice の base offset。stride は element 単位。
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # M/D は [B*H, SEQ_LEN] として保存されているので、その batch-head slice offset を計算する。
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Q pointer をこの batch/head slice の先頭へ進める。
    Q += offset_batch_head
    # K pointer をこの batch/head slice の先頭へ進める。
    K += offset_batch_head
    # V pointer をこの batch/head slice の先頭へ進める。
    V += offset_batch_head
    # dO pointer をこの batch/head slice の先頭へ進める。
    dO += offset_batch_head
    # dQ pointer をこの batch/head slice の先頭へ進める。
    dQ += offset_batch_head
    # dK pointer をこの batch/head slice の先頭へ進める。
    dK += offset_batch_head
    # dV pointer をこの batch/head slice の先頭へ進める。
    dV += offset_batch_head

    # M pointer をこの batch/head slice の先頭へ進める。
    M += offset_batch_head_seq
    # D pointer をこの batch/head slice の先頭へ進める。
    D += offset_batch_head_seq

    # head_dim 方向の offset vector。
    offs_dim = tl.arange(0, HEAD_DIM)

    # grid 第 0 軸を query block index として使っている。
    index_block_kv = tl.program_id(0)

    # この dQ kernel が担当する query block の開始位置。
    start_q = index_block_kv * BLOCK_Q
    # query token offset vector。
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Q block を load する。形状は [BLOCK_Q, HEAD_DIM]。
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    # dQ accumulator を float32 で初期化する。
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    # dO block を load する。形状は [BLOCK_Q, HEAD_DIM]。
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    # M は forward で保存した logsumexp。shape は [BLOCK_Q]。
    M_block = tl.load(M + offs_q)
    # broadcasting 用に [BLOCK_Q, 1] へ変形する。
    M_block = M_block[:, None]

    # key/value block 内の相対 offset。
    offs_kv = tl.arange(0, BLOCK_KV)

    # K を転置 block として読む pointer。形状は [HEAD_DIM, BLOCK_KV]。
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    # V も転置 block として読む pointer。形状は [HEAD_DIM, BLOCK_KV]。
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    # D_i = sum(O_i * dO_i)。shape は [BLOCK_Q]。
    Di = tl.load(D + offs_q)

    # 現在処理中の key/value block の開始位置。
    curr_kv = 0
    # K/V block の数。注意: SEQ_LEN が BLOCK_KV で割り切れる前提。
    num_steps = SEQ_LEN // BLOCK_KV
    # 全 key/value block を走査して dQ を累積する。
    for blk_idx in range(num_steps):
        # K^T block を load する。形状は [HEAD_DIM, BLOCK_KV]。
        K_T_block = tl.load(kT_ptrs)
        # V^T block を load する。形状は [HEAD_DIM, BLOCK_KV]。
        V_T_block = tl.load(vT_ptrs)
        # score S = Q @ K^T * scale。形状は [BLOCK_Q, BLOCK_KV]。
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        # P = exp(S - logsumexp)。forward の softmax 確率を再構成する。
        P_block = tl.math.exp(QK_block - M_block)

        # causal attention なら未来 key を 0 にする。
        if STAGE == 3:
            # 現在の KV block の絶対 key offset。
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            # True は attend 可能な位置。
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            # mask された P を 0 にする。
            P_block = tl.where(mask_block, P_block, 0.0)

        # dP = dO @ V^T。shape は [BLOCK_Q, BLOCK_KV]。
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        # softmax backward: dS = P * (dP - D)。D は row-wise dot(P, dP) と同値。
        dS_block = P_block * (dP_block - Di[:, None])
        # dS を fp16 にして次の tl.dot の Tensor Core 利用を狙う。
        dS_block = dS_block.to(tl.float16)
        # dQ += scale * dS @ K。K_T_block を転置して [BLOCK_KV, HEAD_DIM] に戻す。
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # 次の KV block の開始位置へ進む。
        curr_kv += BLOCK_KV
        # K_T pointer を sequence 方向に BLOCK_KV 進める。
        kT_ptrs += BLOCK_KV * stride_seq
        # V_T pointer を sequence 方向に BLOCK_KV 進める。
        vT_ptrs += BLOCK_KV * stride_seq

    # dQ の保存先 pointer。形状は [BLOCK_Q, HEAD_DIM]。
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # dQ accumulator を global memory に store する。
    tl.store(dQ_block_ptrs, dQ_block)


# dK と dV を計算する backward kernel。
# K/V block を固定し、全 Q block を走査して dK/dV を累積する。
@triton.jit
def _attn_bwd_dk_dv(
    # Q pointer。
    Q,
    # K pointer。
    K,
    # V pointer。
    V,
    # softmax scale。
    softmax_scale,
    # upstream gradient dO pointer。
    dO,
    # dQ pointer。この kernel では実質使わないが signature を揃えている。
    dQ,
    # dK output pointer。
    dK,
    # dV output pointer。
    dV,
    # forward で保存した logsumexp M pointer。
    M,
    # D pointer。
    D,
    # tensor の batch stride。
    stride_batch,
    # tensor の head stride。
    stride_head,
    # tensor の sequence stride。
    stride_seq,
    # tensor の head_dim stride。
    stride_dim,
    # number of heads。
    NUM_HEADS,
    # sequence length。
    SEQ_LEN,
    # query block size。
    BLOCK_Q: tl.constexpr,
    # key/value block size。
    BLOCK_KV: tl.constexpr,
    # head dimension。
    HEAD_DIM: tl.constexpr,
    # causal/non-causal stage。
    STAGE: tl.constexpr,
):
    # grid 第 2 軸は batch-head index。
    index_batch_head = tl.program_id(2)
    # batch index を復元する。
    index_batch = index_batch_head // NUM_HEADS
    # head index を復元する。
    index_head = index_batch_head % NUM_HEADS
    # batch/head slice の base offset。
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # M/D 用の [B*H, SEQ_LEN] offset。
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Q pointer をこの batch/head slice の先頭へ進める。
    Q += offset_batch_head
    # K pointer をこの batch/head slice の先頭へ進める。
    K += offset_batch_head
    # V pointer をこの batch/head slice の先頭へ進める。
    V += offset_batch_head
    # dO pointer をこの batch/head slice の先頭へ進める。
    dO += offset_batch_head
    # dQ pointer をこの batch/head slice の先頭へ進める。
    dQ += offset_batch_head
    # dK pointer をこの batch/head slice の先頭へ進める。
    dK += offset_batch_head
    # dV pointer をこの batch/head slice の先頭へ進める。
    dV += offset_batch_head

    # M pointer をこの batch/head slice の先頭へ進める。
    M += offset_batch_head_seq
    # D pointer をこの batch/head slice の先頭へ進める。
    D += offset_batch_head_seq

    # head_dim 方向の offset vector。
    offs_dim = tl.arange(0, HEAD_DIM)

    # grid 第 0 軸は key/value block index。
    index_block_kv = tl.program_id(0)
    # この kernel instance が担当する key/value block の開始位置。
    start_kv = index_block_kv * BLOCK_KV

    # key/value token offset vector。
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    # dV accumulator。形状は [BLOCK_KV, HEAD_DIM]。
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    # dK accumulator。形状は [BLOCK_KV, HEAD_DIM]。
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # K block を load する。固定した K/V block は inner loop 中ずっと保持する。
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    # V block を load する。
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    # query block 内の相対 offset。
    offs_q = tl.arange(0, BLOCK_Q)

    # Q を転置 block として読む pointer。形状は [HEAD_DIM, BLOCK_Q]。
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    # dO を通常 block として読む pointer。形状は [BLOCK_Q, HEAD_DIM]。
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # 現在処理中の query block の開始位置。
    curr_q = 0
    # query block の数。注意: SEQ_LEN が BLOCK_Q で割り切れる前提。
    num_steps = SEQ_LEN // BLOCK_Q
    # 全 query block を走査して dK/dV を累積する。
    for blk_idx in range(num_steps):
        # Q^T block を load する。形状は [HEAD_DIM, BLOCK_Q]。
        qT_block = tl.load(qT_ptrs)
        # 現在の query block の絶対 offset。
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        # forward で保存した logsumexp を読む。shape は [BLOCK_Q]。
        m = tl.load(M + offs_q)

        # QK_T_block = K @ Q^T * scale。これは score 行列 S の転置 [BLOCK_KV, BLOCK_Q]。
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # P^T = exp(S^T - logsumexp)。softmax probability の転置を再構成する。
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        # causal attention なら未来方向を mask する。
        if STAGE == 3:
            # True は attend 可能な位置。形状は [BLOCK_KV, BLOCK_Q]。
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )
            # mask された probability を 0 にする。
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        # dO block を load する。形状は [BLOCK_Q, HEAD_DIM]。
        dO_block = tl.load(dO_ptrs)
        # dV += P^T @ dO。形状は [BLOCK_KV, HEAD_DIM]。
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # D_i = sum(O_i * dO_i) を読む。shape は [BLOCK_Q]。
        Di = tl.load(D + offs_q)

        # dP^T = V @ dO^T。形状は [BLOCK_KV, BLOCK_Q]。
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # dS^T = P^T * (dP^T - D)。softmax backward の転置版。
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        # fp16 に cast して tl.dot の Tensor Core 利用を狙う。
        dS_T_block = dS_T_block.to(tl.float16)

        # dK += scale * dS^T @ Q。tl.trans(qT_block) は [BLOCK_Q, HEAD_DIM]。
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # 次の query block へ進む。
        curr_q += BLOCK_Q
        # Q^T pointer を sequence 方向に BLOCK_Q 進める。
        qT_ptrs += BLOCK_Q * stride_seq
        # dO pointer を sequence 方向に BLOCK_Q 進める。
        dO_ptrs += BLOCK_Q * stride_seq

    # dV の保存先 pointer。形状は [BLOCK_KV, HEAD_DIM]。
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # dV accumulator を store する。
    tl.store(dV_block_ptrs, dV_block)

    # dK の保存先 pointer。形状は [BLOCK_KV, HEAD_DIM]。
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # dK accumulator を store する。
    tl.store(dK_block_ptrs, dK_block)


# torch.autograd.Function を継承し、forward/backward を custom Triton kernel で実装する。
class TritonAttention(torch.autograd.Function):

    # forward は autograd graph の順伝播。ctx に backward 用情報を保存する。
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # Q/K の head_dim を取り出す。
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        # V の head_dim を取り出す。
        HEAD_DIM_V = V.shape[-1]

        # Q の shape から batch, heads, sequence, head_dim を取り出す。
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        # Q/K/V の head_dim が一致することを要求する。
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # 出力 O を Q と同じ shape/dtype/device で確保する。
        O = torch.empty_like(Q)
        # causal=True なら STAGE=3、non-causal なら STAGE=1。
        stage = 3 if causal else 1

        # grid は Triton kernel の program instance 配置。
        # 第 0 軸: query block 数。第 1 軸: batch*head。第 2 軸: 未使用。
        grid = lambda args: (
            # triton.cdiv は ceil division。SEQ_LEN / BLOCK_SIZE_Q の切り上げ。
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            # batch と head を flatten した program 数。
            BATCH_SIZE * NUM_HEADS,
            # 3D grid の第 2 軸。この forward では 1 固定。
            1,
        )

        # M は backward 用 logsumexp。dtype は数値安定性のため float32。
        M = torch.empty(
            # shape は [batch, heads, sequence]。
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # Triton kernel launch。_attn_fwd[grid](...) という構文で GPU kernel を起動する。
        _attn_fwd[grid](
            # Query tensor。
            Q=Q,
            # Key tensor。
            K=K,
            # Value tensor。
            V=V,
            # attention score scale。
            softmax_scale=softmax_scale,
            # backward 用 logsumexp 保存先。
            M=M,
            # output tensor。
            O=O,
            # Q の stride 群。
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            # K の stride 群。
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            # V の stride 群。
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            # O の stride 群。
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            # batch size。
            BATCH_SIZE=Q.shape[0],
            # number of heads。
            NUM_HEADS=Q.shape[1],
            # sequence length。
            SEQ_LEN=Q.shape[2],
            # head dimension。
            HEAD_DIM=HEAD_DIM_K,
            # causal/non-causal stage。
            STAGE=stage,
        )

        # backward で必要な tensor を保存する。
        ctx.save_for_backward(Q, K, V, O, M)
        # grid lambda を ctx に保存する。ただしこの backward では直接使っていない。
        ctx.grid = grid
        # softmax scale を backward 用に保存する。
        ctx.softmax_scale = softmax_scale
        # head_dim を backward 用に保存する。
        ctx.HEAD_DIM = HEAD_DIM_K
        # causal flag を backward 用に保存する。
        ctx.causal = causal
        # autograd の forward 出力として O を返す。
        return O

    # backward は upstream gradient dO から dQ, dK, dV を計算する。
    @staticmethod
    def backward(ctx, dO):
        # forward で保存した tensor を取り出す。
        Q, K, V, O, M = ctx.saved_tensors

        # dO は contiguous であることを要求する。
        assert dO.is_contiguous()
        # Q/K/V/O/dO が同じ stride であることを要求する。
        # backward kernel の pointer 計算が単一 stride 群を使うため。
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        # dQ を確保する。
        dQ = torch.empty_like(Q)
        # dK を確保する。
        dK = torch.empty_like(K)
        # dV を確保する。
        dV = torch.empty_like(V)

        # shape から batch, heads, sequence を取り出す。
        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        # backward kernel の launch 設定。固定値としている。
        NUM_WARPS, NUM_STAGES = 4, 3
        # micro/macro block size。dQ と dK/dV で役割を入れ替える。
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        # preprocess は query macro block ごと、batch-head ごとに起動する。
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        # D は D_i = sum(O_i * dO_i) を保存する tensor。
        D = torch.empty_like(M)

        # D を計算する前処理 kernel を起動する。
        _attn_bwd_preprocess[preprocess_grid](
            # forward output。
            O=O,
            # upstream gradient。
            dO=dO,
            # D の保存先。
            D=D,
            # sequence length。
            SEQ_LEN=SEQ_LEN,
            # query block size。
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            # head dimension。
            HEAD_DIM=ctx.HEAD_DIM,
        )

        # backward main kernels の grid。
        # 第 0 軸: block index、第 1 軸: ここでは 1、第 2 軸: batch-head。
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        # causal=True なら STAGE=3、non-causal なら STAGE=1。
        stage = 3 if ctx.causal else 1

        # dK/dV kernel を起動する。K/V block を固定して Q block を走査する。
        _attn_bwd_dk_dv[grid](
            # Query tensor。
            Q=Q,
            # Key tensor。
            K=K,
            # Value tensor。
            V=V,
            # softmax scale。
            softmax_scale=ctx.softmax_scale,
            # upstream gradient。
            dO=dO,
            # dQ pointer。ここでは実質未使用。
            dQ=dQ,
            # dK output。
            dK=dK,
            # dV output。
            dV=dV,
            # logsumexp。
            M=M,
            # D_i。
            D=D,
            # stride 群。Q/K/V/O/dO が同じ stride である前提。
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            # number of heads。
            NUM_HEADS=NUM_HEADS,
            # sequence length。
            SEQ_LEN=SEQ_LEN,
            # query block は micro size。
            BLOCK_Q=BLOCK_SIZE_MICRO,
            # key/value block は macro size。
            BLOCK_KV=BLOCK_SIZE_MACRO,
            # head dimension。
            HEAD_DIM=ctx.HEAD_DIM,
            # causal/non-causal stage。
            STAGE=stage,
            # program あたり warp 数。
            num_warps=NUM_WARPS,
            # pipeline stage 数。
            num_stages=NUM_STAGES,
        )

        # dQ kernel を起動する。Q block を固定して K/V block を走査する。
        _attn_bwd_dq[grid](
            # Query tensor。
            Q=Q,
            # Key tensor。
            K=K,
            # Value tensor。
            V=V,
            # softmax scale。
            softmax_scale=ctx.softmax_scale,
            # upstream gradient。
            dO=dO,
            # dQ output。
            dQ=dQ,
            # dK pointer。ここでは実質未使用。
            dK=dK,
            # dV pointer。ここでは実質未使用。
            dV=dV,
            # logsumexp。
            M=M,
            # D_i。
            D=D,
            # stride 群。
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            # number of heads。
            NUM_HEADS=NUM_HEADS,
            # sequence length。
            SEQ_LEN=SEQ_LEN,
            # query block は macro size。
            BLOCK_Q=BLOCK_SIZE_MACRO,
            # key/value block は micro size。
            BLOCK_KV=BLOCK_SIZE_MICRO,
            # head dimension。
            HEAD_DIM=ctx.HEAD_DIM,
            # causal/non-causal stage。
            STAGE=stage,
            # program あたり warp 数。
            num_warps=NUM_WARPS,
            # pipeline stage 数。
            num_stages=NUM_STAGES,
        )

        # forward の入力は Q, K, V, causal, softmax_scale なので、対応する gradient を返す。
        # causal と softmax_scale は tensor ではないため None。
        return dQ, dK, dV, None, None


# correctness test 用の関数。PyTorch reference と Triton 実装を forward/backward で比較する。
def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    # Q tensor を cuda 上に確保し、正規分布で初期化して gradient を有効化する。
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    # K tensor を cuda 上に確保し、正規分布で初期化して gradient を有効化する。
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    # V tensor を cuda 上に確保し、正規分布で初期化して gradient を有効化する。
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    # attention score scale。scaled dot-product attention の 1/sqrt(d) に対応する。
    softmax_scale = 1 / (HEAD_DIM**0.5)
    # backward 比較用の upstream gradient。
    dO = torch.randn_like(Q)

    # PyTorch reference 実装。ここでは P = softmax(QK^T) を明示的に materialize する。
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    # attention score を計算する。shape は [B, H, SEQ_LEN, SEQ_LEN]。
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    # causal=True なら上三角を -inf にする。
    if causal:
        # MASK == 0 の位置は未来 token なので attend できない。
        P[:, :, MASK == 0] = float("-inf")
    # softmax を fp32 で計算し、最後に half へ落とす。
    P = torch.softmax(P.float(), dim=-1).half()
    # O = P @ V を計算する。
    ref_O = torch.matmul(P, V)
    # PyTorch autograd で reference gradient を計算する。
    ref_O.backward(dO)
    # V.grad を退避し、次の backward のために clear する。
    ref_dV, V.grad = V.grad.clone(), None
    # K.grad を退避し、clear する。
    ref_dK, K.grad = K.grad.clone(), None
    # Q.grad を退避し、clear する。
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Triton 実装の forward を呼ぶ。
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    # Triton custom backward を呼ぶ。
    tri_out.backward(dO)
    # Triton 側 dV を退避し、clear する。
    tri_dV, V.grad = V.grad.clone(), None
    # Triton 側 dK を退避し、clear する。
    tri_dK, K.grad = K.grad.clone(), None
    # Triton 側 dQ を退避し、clear する。
    tri_dQ, Q.grad = Q.grad.clone(), None

    # 相対誤差 tolerance。fp16 attention では bitwise 一致は期待しない。
    rtol = 0.0
    # 絶対誤差 tolerance。
    atol = 1e-2
    # forward 出力を比較する。
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    # dK を比較する。
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    # dV を比較する。
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    # dQ を比較する。
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


# このファイルを直接実行した場合だけ test_op を走らせる。
if __name__ == "__main__":
    # causal attention の forward/backward を大きめ shape で検証する。
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    # non-causal attention の forward/backward を同じ shape で検証する。
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    # すべての assert が通れば PASSED を表示する。
    print("PASSED")
