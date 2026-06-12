# Module 17: torch.compile と custom Triton の境界

## 位置づけ

compiler 自動融合で十分な箇所と、手書き Triton kernel が必要な箇所を分ける。

この教材では、新しい用語が初めて重要になる章の先頭で定義します。後で再確認したい場合は [`docs/glossary.md`](../glossary.md) を見てください。

## 新出用語

| 用語 | この章での意味 |
|---|---|
| torch.compile | PyTorch program を graph 化し、backend compiler で最適化する API。 |
| graph capture | Python 実行を演算 graph として捕捉すること。 |
| Inductor | PyTorch 2 系の compiler backend。Triton kernel を生成する場合がある。 |
| custom Triton kernel | ユーザーが直接書いた Triton kernel。 |
| fusion boundary | compiler が演算をまたいで融合できなくなる境界。 |
| fallback | compiler や custom kernel が使えない場合に PyTorch 実装へ戻す経路。 |

## 読みながら確認すること

- この章の用語を、自分の kernel 設計上の decision に対応づける。
- PyTorch で見える抽象概念と、Triton kernel 内で制御する概念を分ける。
- 速さだけでなく、どの memory 階層・実行単位・数値誤差が支配的かを言語化する。

## 次にやる演習

[`exercises/17_torch_compile_and_custom_triton.md`](../../exercises/17_torch_compile_and_custom_triton.md) を解いてください。演習では、この docs で導入した用語を使って、実装または観察結果を説明します。

## 解答・確認用 lesson

演習後に [`lessons/17_torch_compile_and_custom_triton.py`](../../lessons/17_torch_compile_and_custom_triton.py) を実行してください。この lesson がこの module の標準解答・確認スクリプトです。

## 深掘り用 reference docs

- [`docs/reference/02_triton_programming_model.md`](../reference/02_triton_programming_model.md)

## Navigation

前: [`docs/modules/16_bottleneck_lab_and_profiler.md`](16_bottleneck_lab_and_profiler.md) / 次: [`docs/modules/18_ir_ptx_sass_inspection.md`](18_ir_ptx_sass_inspection.md)
