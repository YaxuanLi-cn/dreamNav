unzip University-Release.zip

# 创建新文件夹 tours
mkdir -p tours

cp -r University-Release/train/drone/* tours/
cp -r University-Release/test/gallery_drone/* tours/

rm -rf University-Release/
rm -rf University-Release.zip


hf download --repo-type dataset YaxuanLi/PairUAV --local-dir .

tar -xvf train.tar -C .
tar -xvf test.tar -C .

rm -rf train.tar
rm -rf test.tar
rm -rf .cache