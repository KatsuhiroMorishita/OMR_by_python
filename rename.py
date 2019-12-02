# スキャンした画像に出席番号順で名前を付け替える
# created: 2017-10-04
# license: MIT
import glob
import shutil
import sys
import os
from natsort import natsorted
import scoring as sc


def rename_default():
    """ 画像ファイル名を使って、ファイル名を書き換える
    画像名が既にソートされている場合に利用してください。
    """
    files = glob.glob("origin/*.jpg")  # ファイル一覧を取得
    files = natsorted(files)       # 名前順にソート

    for i in range(len(files)):
        file = files[i]
        if "no" in file:
            continue
        new_name = "source/no{0}.jpg".format(i+1)
        shutil.copyfile(file, new_name)


def rename_with_mark():
    """ マーク内容を使って、ファイル名を書き換える
    """
    # 設定の読み込み
    setting_dict = sc.read_setting("setting.txt")
    th = int(setting_dict["threshold_mark"])    # マークの2値化の閾値

    # 画像ファイル名の一覧取得
    fnames = glob.glob("origin/*.jpg")   # 解答の画像ファイル名を取得
    fnames = natsorted(fnames)   # ファイル名でソート。標準関数のsorted()よりも、デバッグしやすい

    # 画像から番号を取得しつつ、ファイルを別名でコピー
    for fname in fnames:
        _id = sc.get_number_from_marks(fname, 10, 2, threshold_mark=th)    # マークシートから出席番号を取得
        print(fname, _id)
        new_name = "source/no{0}.jpg".format(_id)
        if os.path.exists(new_name):
            print("same name already exists.")
        shutil.copyfile(fname, new_name)


def main():
    if "-m" in sys.argv:
        rename_with_mark()
    else:
        rename_default()


if __name__ == "__main__":
    main()