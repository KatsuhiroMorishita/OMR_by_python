# スキャンした画像に出席番号順で名前を付け替える
# created: 2017-10-04
# license: MIT
import glob
import shutil

files = glob.glob("*.jpg")  # ファイル一覧を取得
files = sorted(files)       # 名前順にソート

for i in range(len(files)):
    file = files[i]
    if "no" in file:
        continue
    new_name = "no{0}.jpg".format(i+1)
    shutil.copyfile(file, new_name)
