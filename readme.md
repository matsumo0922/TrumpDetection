# 使い方
OpenCVとJavaがインストールされたWindowsPCで以下のコマンドを実行します

```commandline
java -jar TrumpDetection-all.jar
```

<p>プログラムが起動し、<code>Enter the image path...</code> と表示されるので、処理したい画像のパス、または処理したい画像群のディレクトリへのパスを入力します。すると自動で処理が開始されるので、終了まで待つだけです。</p>
<p>画像へのパスを指定した場合は、元画像が保存されているディレクトリに-outputという文字列を付け足した画像ファイルが出力されます。ディレクトリへのパスを指定した場合は、ディレクトリ内にoutputというディレクトリが生成され、その中に一括して画像が出力されます。この際のファイル名はもの画像と同じファイル名です。</p>
<p><code>--debug</code>オプションを指定して実行すると、処理途中の画像を出力します。</p>

|元画像|検出結果|
|---|---|
|<img src="https://user-images.githubusercontent.com/56629437/173217056-0c5724d3-a201-42c1-a97c-dca2f5aa367e.jpg" height="324px">|<img src="https://user-images.githubusercontent.com/56629437/173217057-aae230d7-2a5e-4ce5-93c2-a167650f6e42.jpg" height="324px">|
