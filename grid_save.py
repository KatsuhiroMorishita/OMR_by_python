# 画像にグリッドを表示させる
# 設定値を決めるために使う
# created: 2019-06-10
# lisence: MIT
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

im = Image.open('source/no1.jpg')
print(im.size)
X, Y = im.size
draw = ImageDraw.Draw(im)

#フォントの設定(フォントファイルのパスと文字の大きさ)
font = ImageFont.truetype(r"C:\Windows\Fonts\meiryob.ttc", 35)

# 横軸の描画
for y in range(0, Y, 100):
    draw.line(((0, y), (X, y)), fill=(255, 0, 0))
    draw.text((50, y+10), "{}".format(y), fill=(0, 0, 0), fornt=font)

# 縦軸の描画
for x in range(0, X, 100):
    draw.line(((x, 0), (x, Y)), fill=(0, 255, 0))
    draw.text((x+10, 50), "{}".format(x), fill=(0, 0, 0), fornt=font)

# 保存と表示
im.save("grid.jpg")
plt.imshow(im)
plt.show()