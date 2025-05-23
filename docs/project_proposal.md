# AIによるリアルタイムVJ/DJシステム：要件定義とプロジェクト提案

## はじめに: プロジェクトの目的
本プロジェクトは、**Variational Autoencoder (VAE)** を中心としたリアルタイムVJ/DJシステムの開発を目指します。視覚と音響の同期をAIで自動制御し、従来の複雑なシェーダ操作を不要にすることで、直感的でダイナミックなライブ表現を実現します。ユーザーはGitHubからクローンするだけでシステムを即座に利用可能であり、主な利用シーンは小規模クラブでのライブパフォーマンスです。具体的には、**VAEのリアルタイム映像出力（60fps）**をGLSLシェーダーと統合し、MIDIコントローラやiPadによる直感的な操作性を活かした映像演出を行います。

## 1. VAE + GLSLによるリアルタイム映像生成手法
**参考映像の再現**: 提案システムは、YouTube上の参考映像と類似する表現力を目指し、**VAE**を用いた画像生成と**GLSLシェーダー**によるエフェクトを組み合わせます。VAEは学習データに類似した新規画像フレームを次々と生成でき、**連続した潜在空間の補間**によって滑らかな映像変化を実現します。これにGLSLでリアルタイム処理されるフィルタ効果や歪みエフェクトを重ねることで、抽象的かつ高品質なビジュアルを60fpsで出力します。

**VAEモデル構成**: VAEは**Encoder-Decoder構造**で訓練し、潜在空間に音像を対応付けます。潜在次元は表現力とリアルタイム性のバランスから適切に選定します（例: 64次元程度）。VAEはオーディオ入力に直接反応するのではなく、一旦音響特徴（BPM、スペクトル強度など）に基づき**潜在ベクトルの補間パス**を計算し、それに沿って次々と画像フレームを生成します。

## 2. 使用ハードウェアとその仕様
本システムは以下のハードウェアを前提に設計されます。

- **メインPC**: **Alienware Aurora R16**（ハイスペックなゲーミングデスクトップ）を使用します。代表的な構成として、第13世代Intel Core i9プロセッサ + NVIDIA GeForce RTX 40シリーズ GPU（例: RTX 4070以上）搭載モデルを想定します。

- **タブレット端末**: **iPad Pro** をDJ用途で使用します（Algoriddim社の **djay** アプリを想定）。iPad上のdjayからミックスされたオーディオが本システムに入力されます。

- **オーディオインターフェース**: **FOCUSRITE Scarlett Solo (gen. 4)** を使用し、iPadから出力される音声信号をPCに取り込みます。Scarlett Soloは高品位な2イン2アウトIFであり、クラブPA出力とPC音声入力のハブとなります。

- **MIDIコントローラ**: **AKAI APC mini MK2** を操作インターフェースとして用います。これは8x8パッドと各種フェーダーを備えたUSB MIDIコントローラで、ユーザーはこれによりVAE映像のパラメータを直感的に操作できます。

- **出力デバイス**: **プロジェクターまたはLEDスクリーン**を映像出力先とし、**PAスピーカー**を音響出力とします。

## 3. テキスト→動画生成モデル「Sora」の仕様と活用例
本プロジェクトでは、OpenAIが開発した最新のテキストから動画への生成AIモデル **「Sora」** を調査対象とします。**Sora**はテキストプロンプトを入力すると、その記述内容に沿った短い動画クリップを生成できるモデルです。

**活用方法**: Soraは本システムの映像生成に直接リアルタイム利用することは難しいものの、**学習用データセットの拡充**に大きな力を発揮します。例えば、各「作家ペルソナ」の作風を反映した映像素材が不足する場合、Soraに対してその作風を記述したテキストプロンプトを与え、類似したスタイルの人工動画を生成させることができます。

## 4. DevOps/MLOps基盤設計 (CI/CD・学習パイプライン)
システム開発とモデル学習を効率的に進め、再現性と継続的な改良を保証するために、**DevOps/MLOps基盤**を構築します。最低限の構成要素として以下を候補とします。

- **コンテナ環境 (Docker)**: 開発環境・推論環境をDockerイメージで定義します。
- **CI/CD (GitHub Actions)**: GitHubリポジトリに対してCI/CDパイプラインを設定します。
- **機械学習パイプライン・実験管理**: モデル学習やチューニングの効率化のため、**MLflow** 等のプラットフォームを用います。
- **データ管理と学習パイプライン**: 学習用データ（画像・動画クリップ）はGit LFSやDVC(Data Version Control)を利用して管理します。

## 5. VAE学習対象「作家ペルソナ」10名のプロファイルとデータ戦略
本システムではVAEによる映像生成のバリエーションを豊かにするため、**10名の著名アーティストの作風を「ペルソナ」として設定**し、それぞれに合わせたスタイルで映像を生成できるようにします。

1. **落合 陽一（おちあい よういち）** – *メディアアーティスト / 研究者*  
   **作風**: **「デジタルネイチャー」**すなわち「コンピュータと自然が親和することで再構築される新たな自然環境」をテーマに、自然現象と計算機処理を融合させた幻想的な表現を特徴とします。

2. **村上 隆（むらかみ たかし）** – *現代美術アーティスト*  
   **作風**: **スーパーフラット**スタイルで、極彩色の二次元キャラクターや満面の笑みの花、どぎついまでの色遣いがトレードマークです。

3. **バンクシー（Banksy）** – *ストリートアーティスト*  
   **作風**: バンクシーの作品は**シルエットとステンシル**を特徴とし、モノクロームの人物シルエットと鮮やかな赤などのアクセントカラーを組み合わせた構図が多いです。

4. **チームラボ（teamLab）** – *デジタルアートコレクティブ*  
   **作風**: チームラボの作品は**光の粒子と流れ**が特徴的で、無数の光点が有機的に動き、花や滝、波などの自然モチーフを形成します。

5. **レフィク・アナドル（Refik Anadol）** – *メディアアーティスト*  
   **作風**: アナドル氏の作品は**AIによるデータ可視化**が特徴で、膨大なデータセットをニューラルネットワークで処理し、流動的で抽象的な映像として表現します。

6. **Beeple（ビープル、本名: Mike Winkelmann）** – *デジタルアーティスト*  
   **作風**: Beepleの作品は**3DCGとサイバーパンク的世界観**が特徴で、未来的な都市景観、巨大構造物、ポップカルチャーのアイコンを歪んだ形で描くことが多いです。

7. **池田 亮司（いけだ りょうじ）** – *電子音楽家 / ビジュアルアーティスト*  
   **作風**: 池田氏の作品は**ミニマルでデジタル的な精緻さ**が特徴で、幾何学的なグリッドやドットパターン、データの可視化を思わせる抽象的な映像表現が多いです。

8. **真鍋 大度（まなべ だいと）** – *メディアアーティスト / プログラマー*  
   **作風**: 真鍋氏の作品は**インタラクティブ性とアルゴリズミック**な表現が特徴で、人間の動きに反応する映像や、数学的アルゴリズムに基づく幾何学的パターンが多いです。

9. **三上 晴子（みかみ せいこ）** – *メディアアーティスト*  
   **作風**: 三上氏の作品は**生命と機械の融合**をテーマにしたバイオアートが特徴で、有機的な形状と機械的な動きの対比、生体信号を映像化する表現などが見られます。

10. **草間 彌生（くさま やよい）** – *前衛美術家*  
    **作風**: 草間氏といえば**水玉模様の反復**です。赤地に白のドット、黄地に黒のドットなど強烈な色彩コントラストの水玉が空間全体に広がる作品が多いです。

## 6. システム全体構成図: ソフト連携・操作フロー・音映像の流れ
本節では、ハード・ソフト・インタラクションの流れをまとめた**システム全体構成図**を示します。各コンポーネントの接続関係とデータフローを図にします。

- **音響入力**: iPadから出力された音楽はScarlett Soloでライン入力され、PCのオーディオ解析に渡されます。
- **AI映像生成**: VAEモジュールは直前フレームの潜在ベクトルと新たな音響解析情報から次フレームの潜在ベクトルを計算し、画像を生成します。
- **シェーダー合成**: GLSLシェーダーモジュールでは、VAE生成テクスチャを入力として様々なエフェクト演算を行います。
- **ユーザー操作フロー**: VJ/DJプレイヤーは、片手で音楽ミックスをしながらもう一方の手でAPC miniの操作を行う想定です。

## 7. 学習と蒸留の計画: 高性能サーバとローカル環境への対応
最後に、**VAEモデルの学習フェーズとデプロイ（推論）フェーズ**における環境対応について述べます。

- **高性能サーバでの学習**: 複数GPU搭載のワークステーションやクラウドGPUサーバを用意し、VAEモデルの学習や再学習、チューニングを行います。
- **知識蒸留によるモデル軽量化**: 学習済みモデルが推論に重すぎる場合、**Knowledge Distillation（知識蒸留）**の手法で軽量モデルへ圧縮します。
- **ローカルPCでの最適化推論**: Alienware R16上での実行では、可能な限りGPU（例えばRTX 4070）の性能を引き出します。
- **継続的アップデート**: 学習→蒸留→デプロイの流れは一度きりでなく、継続的に回せるようにします。
