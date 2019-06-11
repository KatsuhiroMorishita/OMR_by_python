#     scoring.py
# purpose: マークシートの採点
# author: Katsuhiro Morishita (morimori.ynct@gmail.com)
# created: 2017-12-02
# lincense: MIT
import cv2
import marking_parser as mp
import glob
import numpy as np
import pandas as pd
import os
import sys


def get_socore_a_problem(correct, answer, mode="normal"):
    """ 1問に対する、正解と解答を比較した結果を数値（0.0 - 1.0）で返す
    正解していれば1.0, 不正解で0.0。部分点がある場合は0.0 - 1.0の値を取る。
    correct: 1D ndarray or list, 正解が入っている
    answer: 1D ndarray or list, 検査したい値が入っている
    """
    if len(correct) != len(answer):             # 解答と正解のベクトルの長さが一致しない場合
        print("--vector length is not equal.--")
        return None
    
    vect = []                                   # まずはベクトルの一致状況を確認
    for i in range(len(correct)):
        c = correct[i]                          # ここでのcとaはベクトルの各要素（0か1）を指す
        a = answer[i]
        vect.append(c == a)
    vect = np.array(vect)                       # 一致している部分だけ1となるベクトルができているはず
    
    if mode == "normal":                        # 正解が1つ又は複数あり、解答がいずれかの正解と一致していれば良い場合
        _c = 1 - np.array(correct)              # 0と1を反転
        _a = np.array(answer)
        if np.dot(vect, correct) > 0 and np.dot(_c, _a) == 0:   # 正解との内積>0かつ正解以外にチェックを入れていなければOK
            return 1
        else:
            return 0
    elif mode == "full match":                  # "該当するもの全てにチェックを入れよ"的な、正解との完全一致で満点の場合（マークをつけないことにも強い意味がある）
        return np.sum(vect) / len(vect)
    else:
        raise ValueError('A mode is unknown. Check your correct_answer.xlsx. ')
    
        
def get_score(corrects, answers):
    """ 1枚のマークシートの解答を正解と比較し、点数を格納した1D arrayを返す
    corrects: list, 正解と採点モードのペアを要素とするリスト
    answers: 2D ndarray or list, 検査したい値が入っている
    """    
    scores = []
    for i in range(len(corrects)):
        c, point, mode = corrects[i]
        a = answers[i]
        score = get_socore_a_problem(c, a, mode) * point
        scores.append(score)
        """
        if i == 2:   # デバッグ用
            print("point", type(point), point)
            print("a", type(a), a)
            print("c", type(c), c)
            print("mode", mode)
            print("score", score)
        """
    return scores

        
def read_correct(fname):
    """ pandasを使って正解を読み込む
    fname: str, 正解が記入されたExcelのファイル名
    """
    correct = []
    
    df =  pd.read_excel(fname)
    temp = df.iloc[0, 1]                # 1行目に選択肢の数の情報が入っている（例1：abcdef, 例2:12345）
    
    for i in range(1, len(df)):         # 2行目以降から正解の情報を読取る
        ans =  df.iloc[i, 1]            # 正解（文字列で与えられる）
        point =  df.iloc[i, 2]          # 配点
        mode =  df.iloc[i, 3]           # 採点モード
        c = np.zeros(len(temp))         # 選択肢の数だけ0を用意
        if isinstance(ans, str):        # 解がない場合に対応
            for k in ans:               # 正解の在る部分だけ1とする（複数の選択肢が正解なケースにも対応）
                if k in temp:
                    index = temp.index(k)
                    c[index] = 1
                else:
                    print("--不明な選択肢が正解に入力されています--")
                    continue
        correct.append((c, point, mode))
    return correct



def read_setting(fname):
    """ 設定を読み込む。返り値は辞書で返す
    """
    param = {}
    with open(fname, "r", encoding="utf-8-sig") as fr:
        lines = fr.readlines()

        for line in lines:
            if "," not in line:
                continue
            line = line.rstrip()    # 右端の空白や改行コードを削除
            param_name, value = line.split(",")
            param[param_name] = value

    return param




def get_number_from_marks(fname, W, H, trim_y=(200,1300), trim_x=(200,300), threshold_mark=150):
    """ 出席番号用のマークから番号を取得する
    trim_y: tuple<int, int>, 縦方向の余白幅。上下
    trim_x: tuple<int, int>, 横方向の余白幅。左右
    threshold_mark: int, 0 <= x <= 255. 解答欄に着いたマークの2値化の閾値。小さいほど濃ゆくないと駄目。
    """
    img = cv2.imread(fname)                        # カラー画像を読み込み
    Y, X, Z = img.shape
    upper = trim_y[0]
    under = trim_y[1]
    left = trim_x[0]
    right = trim_x[1]
    img = img[upper:Y-under, left:X-right]         # 画像の縁と上部を取り除いた画像を作成
    result_i = mp.read_marking(img, W, H, mp.bin_by_red, fname, "i", threshold_mark=threshold_mark)   # 上側の判定結果を取得
    
    try:
        id1 = np.nonzero(result_i[0] == 1)[0][0]
        id2 = np.nonzero(result_i[1] == 1)[0][0]
        nums = [1,2,3,4,5,6,7,8,9,0]
        return 10 * nums[id1] + nums[id2]
    except:
        print("ファイル:" + fname + " から学生番号を読み取ることに失敗しました．", file=sys.stderr)
        return 0



def read(fname, W, H, trim_y=(500,50), trim_x=(50,50), threshold_mark=150):
    """ 指定されたファイルに記載されたマーカーのマーク位置を返す
    (1) 用紙のスキャン画像上部を切り取る用に設計しているが、フォーマットによっては不要だろう。
    (2) ↑での切り取り幅は画像の解像度によって変わる
    W: int, 横方向の選択肢の数
    H: int, 縦方向の設問の数
    trim_y: tuple<int, int>, 縦方向の余白幅。上下
    trim_x: tuple<int, int>, 横方向の余白幅。左右
    threshold_mark: int, 0 <= x <= 255. 解答欄に着いたマークの2値化の閾値。小さいほど濃ゆくないと駄目。
    """
    img = cv2.imread(fname)                        # カラー画像を読み込み
    Y, X, Z = img.shape
    upper = trim_y[0]
    under = trim_y[1]
    left = trim_x[0]
    right = trim_x[1]
    img = img[upper:Y-under, left:X-right]         # 画像の縁と上部を取り除いた画像を作成
    
    # もし、欄が複数に分かれている場合は複数回に分けて（欄の色で渡す二値化関数を分けるか、おおよその位置で画像を切り出して）取得して、それぞれのresultをnp.vstack()で結合すること
    result_l = mp.read_marking(img, W, H, mp.bin_by_blue, fname, threshold_mark=threshold_mark)   # 左側の判定結果を取得
    result_r = mp.read_marking(img, W, H, mp.bin_by_green, fname, threshold_mark=threshold_mark)  # 右側の判定結果を取得
    print("Left:")
    print(result_l)
    print("Right:")
    print(result_r)
    result = np.vstack((result_l, result_r))
    return result





############################################################
# 用紙や画像ファイル名フォーマットに合わせて、以下の関数は調整する必要がある

def get_number(name):
    """ ファイル名から番号を取得する
    とりあえず、"no\d\d.jpg"のようなフォーマットを想定している。番号は1以上であること。
    """
    name = os.path.basename(name)   # フォルダ名を除いたファイル名を取得
    fname, ext = os.path.splitext(name)
    number = fname[2:4]
    if number[-1] == ".":
        number = number[:-1]
    return int(number)
 



def main():
    # 設定の読み込み
    setting_dict = read_setting("setting.txt")
    W = int(setting_dict["W"])    # 横方向の選択肢の数
    H = int(setting_dict["H"])    # 縦方向の設問の数
    th = int(setting_dict["threshold_mark"])    # マークの2値化の閾値
    corrects = read_correct("correct_answer.xlsx")   # 正解の読み込み
    df = pd.DataFrame()
    fnames = glob.glob("source/*.jpg")   # 解答の画像ファイル名を取得
    fnames = sorted(fnames)
    if len(fnames) == 0:
        print("sourceフォルダに画像が1枚もありません。")
        exit()
    answers_array = []
    save_name = "students answers.npy"  # 学生の解答を保存するファイル名


    # 解答の読み込み
    if "-r" in sys.argv:  # 配点のみ見直したい場合はオプションを付けること
        # 過去に読み込み済みの解答をファイルから読み込み
        answers_array_nd = np.load(save_name)
        answers_array = list(answers_array_nd)
    else:
        # 解答を画像から読み込み
        for fname in fnames:
            answers = read(fname, W, H, threshold_mark=th)   # 解答の読み込み
            answers_array.append(answers)
            print(fname, answers)
        
        answers_array_nd = np.array(answers_array)
        np.save(save_name, answers_array_nd)

    # 採点
    for i in range(len(fnames)):
        fname = fnames[i]
        answers = answers_array[i]    # 格納順とファイル名の順番がズレるとまずいので、画像ファイルの更新には注意
        result = get_score(corrects, answers)    # 点数の取得

        # 学生の番号を取得して、点数と合体
        if "--read-student-id" in sys.argv:
            _id = get_number_from_marks(fname, 10, 2, threshold_mark=th)    # マークシートから出席番号を取得
            series = pd.Series([_id] + result)
        elif "-w" in sys.argv:
            _id1 = get_number(fname)                     # ファイル名から出席番号を取得
            _id2 = get_number_from_marks(fname, 10, 2, threshold_mark=th)   # マークシートから出席番号を取得
            series = pd.Series([_id1, _id2] + result)
        else:
            _id = get_number(fname)                  # ファイル名から出席番号を取得
            series = pd.Series([_id] + result)

        df = df.append(series, ignore_index = True)
    
    # 整理と保存
    offset = 1
    df = df.sort_values(by=[0], ascending=True)            # 出席番号でソート
    df = df.reset_index(drop=True)                         # インデックスを振り直す
    df = df.rename(index=int, columns={0: "Student Num."}) # カラム名を書き換え
    if "-w" in sys.argv:                                   # ファイル名と画像の両方から学生の番号を読んだら
        df = df.rename(index=int, columns={1: "Student Num. from Marks"}) # カラム名を書き換え
        offset += 1
    #df.to_excel("scoring_result.xlsx")   # 保存されない？
    df.to_csv("scoring_result.csv", index=False)
    
    # 正解率を計算して保存する
    correct_rate = []
    for i in range(len(corrects)):
        c, point, mode = corrects[i]
        total = np.sum(df[i + offset])  # 設問毎に、全員の点数の合計を取得
        rate = float("nan")
        if point > 0.0:                 # 0のときは配点していない==計算対象外
            rate = total / (len(df[i + offset]) * point)
        correct_rate.append(rate)
    correct_rate = np.array(correct_rate)
    np.savetxt("correct rate.txt", correct_rate)
    
    # 解答のパターンを掴むために、全解答を重ねて保存する
    superposition_answer = answers_array[0]
    for i in range(1, len(answers_array)):
        superposition_answer = superposition_answer + answers_array[i]
    np.savetxt("superposition answer.csv", superposition_answer, delimiter=',')
    
    print("end")
        
if __name__ == "__main__":
    main()
