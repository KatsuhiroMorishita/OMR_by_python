#    marking_parser
# purpose: マークシートの読み取り
# author: Katsuhiro Morishita (morimori.ynct@gmail.com)
# created: 2017-12-02
# lincense: MIT
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


threshold_mark = 200   # 値が小さいほど、マークの色が濃ゆくないと識別してくれない

def show(img, color=None):
    """ 画像を表示する
    """
    show_img = img.copy()
    if color is not None:
        show_img = cv2.cvtColor(img, color) 
    plt.figure(figsize=(30,15))
    plt.imshow(show_img)
    plt.show()


def img2bin(img, thresh, max_pixel):
    """ 画像を二値化する
    img: 2D ndarray, グレースケール画像
    https://www.blog.umentu.work/python-opencv3%E3%81%A7%E7%94%BB%E5%83%8F%E3%81%AE%E7%94%BB%E7%B4%A0%E5%80%A4%E3%82%92%E4%BA%8C%E5%80%A4%E5%8C%96%E3%81%97%E3%81%A6%E5%87%BA%E5%8A%9B/
    """
    ret, img_dst = cv2.threshold(img,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)
    #print("二値化画像")
    #show(img_dst, cv2.COLOR_GRAY2RGB)
    return img_dst


def mor(img, kernel_size=5, times=1, mode="膨張収縮"):
    """ 膨張・収縮
    img: 2D ndarray, 2値画像
    mode: str, 膨張と収縮の指示。"膨張収縮"だと膨張を先に行う。"収縮膨張膨張"だと、収縮を先にやって、次に膨張を2回行う
    times: int, mode="膨張収縮"のときにtimes=2の場合、膨張が2回実施された後に収縮が2回実施される
    http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_mor = img[:, :]   # copy image
    
    while len(mode) >= 2:
        _mode = mode[:2]
        mode = mode[2:]

        if "収縮" in _mode:     # 黒の領域が増える
            for _ in range(times):
                img_mor = cv2.erode(img_mor, kernel, iterations=1)

        if "膨張" in _mode:      # 白の領域が増える
            for _ in range(times):
                img_mor = cv2.dilate(img_mor, kernel, iterations=1)
    return img_mor


def get_xy(img, n=2, fname=""):
    """ 枠を抽出する
    img: 2値画像
    n: int, 抽出する座標数（四角い枠が1つなら、2）
    fname: str, ファイル名（デバッグ用）
    """
    # 横方向の走査
    total_y = [np.sum(y)  for y in img]
    total_y = np.array(total_y)
    total_y2 = 1 - total_y / np.max(total_y)  # 最大1に正規化したものを1から引く
    total_y2 = total_y2 / np.max(total_y2)   # 最大1に正規化
    total_y2 = [total_y2[i] if total_y2[i] > 0.5 else 0 for i in range(len(total_y2))]       # 小さい値を消す
    index_y = [i for i in range(len(total_y2))  if total_y2[i-1] == 0 and total_y2[i] > 0]   # 立ち上がりを探す
    area = [np.sum(total_y2[i:i + 10]) for i in index_y]
    area[-1] += 0.1        # 2本の線の検出量が全く同じことがあるが、それが後のindex検索に邪魔なので、それを崩す
    #plt.plot(total_y2)    # test for debug
    #plt.show()
    dict_index = {}
    for s, i in zip(area, index_y):
        dict_index[s] = i
    area = sorted(area, reverse=True)
    index_y = [dict_index[s] for s in area[:ｎ]]  # 上位n個を取り出す

    # 縦方向の走査
    img_mor_T = img.T
    total_x = [np.sum(x)  for x in img_mor_T]
    total_x = np.array(total_x)
    total_x2 = 1 - total_x / np.max(total_x)  # 最大1に正規化したものを1から引く
    total_x2 = total_x2 / np.max(total_x2)   # 最大1に正規化
    total_x2 = [total_x2[i] if total_x2[i] > 0.5 else 0 for i in range(len(total_x2))]          #  小さい値を消す
    index_x = [i for i in range(len(total_x2))  if total_x2[i-1] == 0 and total_x2[i] > 0] 
    area = [np.sum(total_x2[i:i + 10]) for i in index_x]
    area[-1] += 0.1        # 2本の線の検出量が全く同じことがあるが、それが後のindex検索に邪魔なので、それを崩す
    #plt.plot(total_x2)    # test for debug
    #plt.show()
    dict_index = {}
    for s, i in zip(area, index_x):
        dict_index[s] = i
    area = sorted(area, reverse=True)
    index_x = [dict_index[s] for s in area[:n]]  # 上位n個を取り出す
    #print(fname, index_x, index_y)    # test for debug

    return index_x, index_y


def get_mark(img, p1, p3, W, H, fname="", img_origin=None):
    """ 指定された枠の中から、マーキングの結果を判定して返す（結果を2Dのndarrayで表す。マークがあれば1、なければ0。）
    W: int, 横方向の選択肢の数
    H: int, 縦方向の設問の数
    fname: str, ファイル名（デバッグ用）
    """
    #print("--searching--", fname, p1, p3)  # for debug

    y0, y1 = p1[1], p3[1]
    x0, x1 = p1[0], p3[0]
    y_step = (y1 - y0) / H   # ここでは浮動小数でないと、計算誤差が溜まるw
    x_step = (x1 - x0) / W
    y_pad = y_step / 4   # 枠線を読みたくないので、若干内側を走査する
    x_pad = x_step / 3
    
    # 各セルの中の黒のピクセル数を集計
    area_arr = []
    for i in range(H):
        area = []
        for j in range(W):
            sx = int(x0 +  x_step * j + x_pad)
            ex = int(x0 +  x_step * (j+1) - x_pad)
            sy = int(y0 +  y_step * i + y_pad)
            ey = int(y0 +  y_step * (i+1) - y_pad)
            img_bit = img[sy : ey, sx : ex]
            black_point = np.count_nonzero(255 - img_bit)  # 0（黒）をカウント（関数自体は0じゃないものをカウントしようとする）
            area.append(black_point)
        area_arr.append(np.array(area))
    
    # マーキングの判定
    area_arr  = np.array(area_arr)      # 生の集計結果
    mean = np.mean(area_arr)
    std = np.std(area_arr)
    mark = np.zeros(area_arr.shape)

    marked_area_ratio = np.sum(area_arr) / (x_step * y_step)  # 塗りつぶされた面積が正味何個分かカウント
    if mean == 0.0 or marked_area_ratio < 0.3:                # マークがない場合は、0解答で返す
        #print("--markless--", fname, p1, p3, marked_area_ratio)
        return mark

    area_arr = (area_arr - mean) / std    # 正規化
    mark[area_arr > 0.9] = 1              # マークされたセルの値を1にセットする（その他は0のまま）。-1を閾値にしたいところだが、難しい・・・
    
    # デバッグ用に検出した結果を画像として保存
    if img_origin is not None:
        import datetime
        import copy

        name = os.path.split(fname)[-1]   # フォルダ名を除く
        name, ext = os.path.splitext(name)

        def save(img4save, key):
            _img = copy.deepcopy(img4save)  # 画像のコピー
            for i in range(H):
                for j in range(W):
                    if mark[i][j] == 1:
                        sx = int(x0 +  x_step * j + x_pad)
                        ex = int(x0 +  x_step * (j+1) - x_pad)
                        sy = int(y0 +  y_step * i + y_pad)
                        ey = int(y0 +  y_step * (i+1) - y_pad)
                        cv2.rectangle(_img, (sx, sy), (ex, ey), (0, 0, 255), 2)
            cv2.imwrite("check/detected_{0}_{1}_x{2}".format(key, name, int(x0 / 100) * 100)  + ".png", _img)
        save(img_origin, "o")
        save(img, "b")
    return mark



def bin_by_gray(img):
    """ グレースケールで二値化した画像を作る
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール画像の作成
    return img2bin(img_gray, 240, 255)


def bin_by_blue(img):
    """ 青色で二値化した画像を作る（カラー部分が黒として残る）
    img: ndarray, BGR画像（OpenCVで読みだした直後のカラー画像はBGR）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([80, 5, 5])   # blue: 120
    upper_hsv = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)   # Threshold the HSV image to get only blue colors
    mask = 255 - mask
    #show(np.array(mask), cv2.COLOR_GRAY2RGB)
    return mask


def bin_by_green(img):
    """ 緑色で二値化した画像を作る（カラー部分が黒として残る）
    img: ndarray, BGR画像（OpenCVで読みだした直後のカラー画像はBGR）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([35, 20, 20])   # green: 65
    upper_hsv = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)   # Threshold the HSV image to get only blue colors
    mask = 255 - mask
    return mask
    
    
def read_marking(img, W, H, bin_func, fname=""):
    """ 画像の中から四角の枠（スキャン画像に対して概ね回転していない事が必須）を検出して、その中のマーキングの結果を取得する
    img: カラー画像（BGR形式）
    W: int, 横方向の選択肢の数
    H: int, 縦方向の設問の数
    bin_func: ２値化する関数
    """
    # 図枠の座標を取得
    img_bin = bin_func(img)   # 回答欄の図枠を読み出すために、二値化
    img_mor = img_bin[:, :]
    img_mor = mor(img_mor, kernel_size=4)                         # ゴミのような点を削除
    img_mor = mor(img_mor, mode="膨張", times=2, kernel_size=2)    # 細かい線が残っているので消す
    #img_mor = mor(img_mor, mode="収縮", times=1, kernel_size=2)   # 細かい線が残っているので消す
    #img_mor = mor(img_mor, mode="膨張", times=1, kernel_size=2)   # 細かい線が残っているので消す
    #show(img_mor, cv2.COLOR_GRAY2RGB)
    xy = get_xy(img_mor, fname=fname)        # 図枠の座標を取得
    if len(xy[0]) == 1 or len(xy[1]) == 1:   # 図枠の検出に失敗したら、return
        print("--lines are not detected.--", fname, xy)
        return None
    p1 = (min(xy[0]), min(xy[1]))            # 枠の左上座標
    p3 = (max(xy[0]), max(xy[1]))            # 枠の右下座標
    
    # マーキングを判定しやすくするために、２値化などの前処理
    img_bin2 = img2bin(img, threshold_mark, 255)   # マーク読み出しのために、鉛筆で塗った後をはっきり黒になるように二値化。第2引数が小さいと、より濃ゆいものが白に分類される。
    kernel = np.ones((3, 3), np.uint8)             # マークの読み取りがしやすいように、ノイズを排除
    for _ in range(2):
        #img_bin2 = cv2.morphologyEx(img_bin2, cv2.MORPH_CLOSE, kernel)
        img_bin2 = mor(img_bin2, kernel_size=5, times=1, mode="収縮膨張")   # この辺は画像によって調整
    for _ in range(2):
        img_bin2 = mor(img_bin2, kernel_size=3, times=2, mode="膨張収縮")

    # マーキングを判定
    result = get_mark(img_bin2, p1, p3, W, H, fname, img_origin=img)
    
    return result



def get_error(a_result):
    """ 各設問毎のエラーを返す
    エラーであれば、Trueが格納されている。
    作っては見たものの、各設問毎に可能なマーク数が1つの場合にしか適用できないのであまり役立たないかも。
    a_result: 2D ndarray, 各設問毎の投票状態が2重の入れ子構造で格納されていること
    """
    vote = np.array([np.sum(x) for x in a_result])     # 各設問に対する投票数を集計
    vote[vote <= 1] = False
    vote[vote >= 2] = True              # 2つ以上解答欄を塗りつぶしていたらエラーとする
    #error_num = np.count_nonzero(vote)
    #print(error_num)
    return vote



############################################################
# 以下は本スクリプトのテスト用の関数
# 
# 用紙フォーマットに合わせて、関数read()は調整する必要がある。
# ただし、scoring.pyを使う場合はそちらで調整すること。
# 
# (1) 関数read()では、用紙のスキャン画像上部を切り取る用に設計しているが、フォーマットによっては不要だろう。
# (2) ↑での切り取り幅は画像の解像度によって変わる

def read(fname, W, H):
    """ 指定されたファイルに記載されたマーカーのマーク位置を返す
    W: int, 横方向の選択肢の数
    H: int, 縦方向の設問の数
    """
    img = cv2.imread(fname)                        # カラー画像を読み込み
    s = img.shape
    img = img[500:s[0]-50, 50:s[1]-50]             # 画像の縁と上部を取り除いた画像を作成
    
    result = read_marking(img, W, H, bin_by_gray, fname)     # 判定結果を取得  
    return result


def main():
    fnames = glob.glob("*.jpg")
    for fname in fnames:
        result = read(fname, 7, 20)
        #print(fname, result)
        #print(fname, get_error(result))
    print("end")

if __name__ == "__main__":
    main()
