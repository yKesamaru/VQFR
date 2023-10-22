# 画像拡大マスター：超解像セットアップシリーズ③ VQFR
超解像シリーズは全部で4編書きます。
1.　MAX Image Resolution Enhancer
2.　GFPGAN
3.　VQFR
4.　1~3の比較

この超解像シリーズでは、いくつかの超解像技術をピックアップして、それぞれの環境構築方法を中心にご紹介します。

今回は、`VQFR`です。

![](https://raw.githubusercontent.com/yKesamaru/VQFR/master/assets/eye_catch.png)

## 出力結果
顔に特化しすぎた学習モデルなためか、一般物の超解像には向いていないようです。
というか、猫などの一般物を入力すると、エラーが出てしまいます。
また、人物写真であっても、顔が小さすぎると、同じ様にエラーが出てしまいます。
ただし認識すれば、4倍の拡大にも耐えられます。
### 元画像
![](https://raw.githubusercontent.com/yKesamaru/VQFR/master/assets/2023-10-22-16-07-09.png)
![](https://raw.githubusercontent.com/yKesamaru/VQFR/master/assets/2023-10-22-16-07-35.png)
### 超解像画像
![](https://raw.githubusercontent.com/yKesamaru/VQFR/master/assets/2023-10-22-16-09-12.png)
![](https://raw.githubusercontent.com/yKesamaru/VQFR/master/assets/2023-10-22-16-09-40.png)

- [画像拡大マスター：超解像セットアップシリーズ③ VQFR](#画像拡大マスター超解像セットアップシリーズ-vqfr)
  - [出力結果](#出力結果)
    - [元画像](#元画像)
    - [超解像画像](#超解像画像)
  - [VQFRとは](#vqfrとは)
    - [論文](#論文)
      - [主なポイント：](#主なポイント)
        - [補足](#補足)
  - [ホスト環境](#ホスト環境)
  - [ローカル環境構築](#ローカル環境構築)
  - [推論の実行](#推論の実行)
    - [引数一覧](#引数一覧)
  - [まとめ](#まとめ)

## VQFRとは
VQFRは、顔復元のためのツールで、TencentARCによって提供されています。

リポジトリには、コードとモデルへのリンクが含まれています。

https://github.com/TencentARC/VQFR

また、以下のデモページでVQFRの動作を確認できます。

https://replicate.com/tencentarc/vqfr?prediction=o4zpz3jbosqkcf5nqzj3yo6ojq

### 論文
VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder

https://arxiv.org/pdf/2205.06803.pdf

#### 主なポイント：
1. **問題設定**: 顔の復元において、入力に忠実な微細な顔の詳細を生成するのは依然として困難である。
2. **VQFRの提案**: VQ技術を利用し、高品質な低レベル特徴バンクを利用してリアルな顔の詳細を回復するVQFRを提案。
3. **ネットワーク設計の改良**: VQコードブックの適切な圧縮パッチサイズの設計と、テクスチャデコーダとメインデコーダを含む並列デコーダの導入を行い、リアルな詳細の生成とアイデンティティの保持を向上させる。
4. **性能**: VQFRは、顔の詳細の復元品質を大幅に向上させ、以前の方法に対する忠実度を保持する。
##### 補足
VQ（Vector Quantization, ベクトル量子化）技術は、連続的なベクトル空間を有限の離散ベクトルの集合にマッピングする方法であり、これによりデータの圧縮と簡単な表現が可能になります。

VQコードブックは、この離散ベクトルの集合を指します。それぞれの離散ベクトルは、コードブックのエントリと呼ばれ、連続的なベクトル空間の特定の領域を表現します。

## ホスト環境
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0

$ nvidia-smi
Fri Oct 20 18:38:04 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:08:00.0  On |                  N/A |
| 41%   38C    P8    13W / 120W |    842MiB /  6144MiB |      5%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

$ inxi -SGm
System:    Host: user Kernel: 5.15.0-86-generic x86_64 bits: 64 Desktop: Gnome 3.36.9 
           Distro: Ubuntu 20.04.6 LTS (Focal Fossa) 
Memory:    RAM: total: 15.55 GiB used: 8.42 GiB (54.2%) 
           RAM Report: permissions: Unable to run dmidecode. Root privileges required. 
Graphics:  Device-1: NVIDIA TU116 [GeForce GTX 1660 Ti] driver: nvidia v: 525.85.12 
           Display: x11 server: X.Org 1.20.13 driver: fbdev,nouveau unloaded: modesetting,vesa resolution: 2560x1440~60Hz 
           OpenGL: renderer: NVIDIA GeForce GTX 1660 Ti/PCIe/SSE2 v: 4.6.0 NVIDIA 525.85.12 
```

## ローカル環境構築

1. **リポジトリのクローン**:
   まずはVQFRのリポジトリをクローンします。
```bash
$ git clone https://github.com/TencentARC/VQFR.git
```

2. **ディレクトリの変更**:
   クローンしたリポジトリのディレクトリに移動します。
```bash
$ cd VQFR
```

3. **仮想環境の作成とアクティベート**:
   Pythonの仮想環境を作成し、アクティベートします。
```bash
$ python3 -m venv .
$ . bin/activate
```

4. **パッケージマネージャのアップデート**:
   最新の`pip`, `setuptools`, `wheel`をインストールします。
```bash
(VQFR) $ pip install -U pip setuptools wheel
```

5. **依存関係のインストール**:
   必要な依存関係をインストールします。
```bash
(VQFR) $ pip install -r requirements.txt
(VQFR) $ VQFR_EXT=True python setup.py develop
(VQFR) $ pip install basicsr
(VQFR) $ pip install facexlib
(VQFR) $ pip install realesrgan
```
6. **モデルのダウンロード**:
    事前学習モデルの場所が非常に分かりにくいです。
    後述の理由で、`VQFR_v2.pth`だけあれば良いと思います。
    これらの事前学習モデルを、`experiments/pretrained_models`に配置して下さい。
   [VQFR_v1-33a1fac5.pth](https://drive.google.com/drive/folders/1lczKYEbARwe27FJlKoFdng7UnffGDjO2)
   [VQFR_v2.pth](https://github.com/TencentARC/VQFR/releases/download/v2.0.0/VQFR_v2.pth)

## 推論の実行
```bash
(VQFR) 
$ python demo.py -i my_images/ -v 2.0 -s 4 -f 0.1
/home/user/bin/VQFR/VQFR/lib/python3.8/site-packages/torchvision/transforms/functional_tensor.py:5: UserWarning: The torchvision.transforms.functional_tensor module is deprecated in 0.15 and will be **removed in 0.17**. Please don't rely on it. You probably just need to use APIs in torchvision.transforms.functional or in torchvision.transforms.v2.functional.
  warnings.warn(
/home/user/bin/VQFR/VQFR/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/user/bin/VQFR/VQFR/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
Processing 739eb804159fa0b021fefc4af026a214.jpg ...
	Tile 1/12
	Tile 2/12
	Tile 3/12
	Tile 4/12
	Tile 5/12
	Tile 6/12
	Tile 7/12
	Tile 8/12
	Tile 9/12
	Tile 10/12
	Tile 11/12
	Tile 12/12
Results are in the [results] folder.

```

### 引数一覧
```bash
Usage: python demo.py -i inputs/whole_imgs -o results -v 2.0 -s 2 -f 0.1 [options]...

  -h                   show this help
  -i input             Input image or folder. Default: inputs/whole_imgs
  -o output            Output folder. Default: results
  -v version           VQFR model version. Option: 1.0. Default: 1.0
  -f fidelity_ratio    VQFRv2 model supports user control fidelity ratio, range from [0,1]. 0 for the best quality and 1 for the best fidelity. Default: 0
  -s upscale           The final upsampling scale of the image. Default: 2
  -bg_upsampler        background upsampler. Default: realesrgan
  -bg_tile             Tile size for background sampler, 0 for no tile during testing. Default: 400
  -suffix              Suffix of the restored faces
  -only_center_face    Only restore the center face
  -aligned             Input are aligned faces
  -ext                 Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

## まとめ
このリポジトリの最終更新日は去年ですが、Issueにあるエラーが未解決のままです。
このため、READMEにあるVQFRv1モデルが使用できません。（実行中にエラーが発生します）
一応このエラーには返答がついているのですが、そのとおりにしても解決しませんでした。
ソースコードに手を入れてデバッグしながら調整しましたが、うまくいきませんでした。
```bash
Processing a.jpeg ...
Segmentation fault (コアダンプ)
```
また、時々GPUをリロードしないと問題が解決できないことがありました。
```bash
sudo rmmod nvidia_uvm
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```
これは私の環境だけで起こるのかもしれません。

また、VQFR_v2.pthモデルがなかなか見つかりませんでしたが、Issueの中にありました。
すこしもったいないリポジトリだな、というのが正直なところです。

ただし、上述のデモページでは、小さな顔の検出や、一般物の超解像が実現できています。

以上です。ありがとうございました。