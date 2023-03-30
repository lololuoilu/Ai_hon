#----------------------
# 必要なライブラリをインポート
#----------------------
# 基礎
import streamlit as st
import openai

import os
import requests
import re
import json
import ast
import io

# データ処理
import numpy as np
import pandas as pd

# 画像にテキスト
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# コネクションエラー
from requests.exceptions import ConnectionError
import time

# 画像の続きを作成する
import base64

CHATGPT_API_KEY = os.environ.get('OPENAI')
# CHATGPT_API_KEY = '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'
openai.api_key = CHATGPT_API_KEY

#----------------------
# 設定入力とストラクチャ
#----------------------
# タイトルとテキスト
st.title('Ai-hon')
st.header('絵本生成アプリ')
st.write('サイドバーで詳細設定をしてください。')
st.write('設定後、生成ボタンを押してください。')


# サイドバー（入力画面）
st.sidebar.header('Input Features')
# st.sidebar.write('背景')
# st.sidebar.write('人物')
# st.sidebar.write('物語のパターン')
# https://www.gizmodo.jp/2016/07/6fictionplot.html
# st.sidebar.write('レイアウト')
# st.sidebar.write('タイトル'）
# chara = st.sidebar.text_input('キャラクター','')
# text_input = st.text_area("物語に欲しい要素を入力してください。", "")
# st.sidebar.write('画像の一部を変更')
# st.sidebar.write('物語を考えてもらう')
# st.sidebar.write('この物語で出力')
# st.sidebar.write('保存')
# text = st.text_input("画像を生成するための文字", "")

style_list = ['絵本', 'イラスト', '絵画', '写実', '幻想的', '印象派']
style_dict = {'絵本':'picture book', 'イラスト':'illustration', '絵画':'painting', '写実':'realistic', '幻想的':'fantastic', '印象派':'impressionism'}

style = st.sidebar.selectbox('作風:',style_list)
style = style_dict[style]

age = st.sidebar.slider('対象年齢', min_value=0, max_value=18, step=1)
# https://happylilac.net/ehon-erabikata-point1-nenreibetubunsetsu.html
pages = st.sidebar.slider('最大ページ数', min_value=8, max_value=16, step=1)


#----------------------
# ChatGPT
#----------------------
def generate_story(pages, age):
    res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content":f"対象年齢が{age}歳向けの絵本の文章を日本語で作成してください。{pages}行、出力しなさい。"
        },

    ],
    )
    return res["choices"][0]["message"]["content"]

def story_line(text):
    res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content":f"{text}を1行づつ、出力してください。"
        },
    ],
    )
    return res["choices"][0]["message"]["content"]

if 'j_list' not in st.session_state: 
    st.session_state.j_list = []
j_list = []

def generate_prompt(text):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":f"{text}を画像生成aiのための、英語のPromptに変換してください。"
                # "content":f"{text}を画像生成aiのための、英語のPromptに1行で変換してください。"

            },
            ],
        )
    return res["choices"][0]["message"]["content"]

#----------------------
# 画像生成
#----------------------
def generate_t2i(text,pages,style,preb_line):

    # 応答設定
    if preb_line == None:
        prompt = f"Generate '{text}' image with a {style} image."
    else:
        prompt = f"Generate '{text}', that is the continuation of this story '{preb_line}', with a {style} image."
    response = openai.Image.create(
                #   prompt = f"以下の文章を{style}風に描画してくだ'さい。'{text}'",# 画像生成に用いる説明文章
                  prompt = prompt,
                  n = 1,                     # 何枚の画像を生成するか
                  size = '256x256',          # 画像サイズ
                  response_format = "url"    # API応答のフォーマット
                )
            
    # API応答から画像URLを指定
    image_url = response['data'][0]['url']
    
    # 画像として取得
    image_data = requests.get(image_url).content
    image_data = Image.open(io.BytesIO(image_data))
    # with open("chat-gpt-generated-image.jpg", "wb") as f:
        # f.write(image_data)
    # return image_url
    return image_data

#----------------------
# 文字入力
#----------------------

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL

def cv2_putText_1(img, text, org, fontFace, fontScale, color):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    draw.text(xy = (x,y), text = text, fill = colorRGB, font = fontPIL)
    # w, h = draw.textsize(text, font = fontPIL)
    # draw.rectangle([(x,y), (x+w,y+h)], outline = (255,0,0), width = 1)
    # draw.ellipse([(x-3,y-3), (x+3,y+3)], None, (255,0,0), 1)
    # imgPIL.show()
    imgCV = pil2cv(imgPIL)
    return imgCV

def with_text(text, img):

    # # 画像を読み込む
    # 画像と同じ高さの白い背景を作成
    # height, width, _ = img.shape[::]
    width, height = img.size

    white_bg = np.zeros((height, width+width, 3), dtype=np.uint8)
    white_bg.fill(255)

    # 画像を白い背景の左側に配置
    white_bg[:, :width, :] = img

    # 文字を入れる
    # fontFace = "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"
    fontFace = "Meiryo"
    # fontFace_cv2 = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 12
    color = (0, 0, 0)
    thickness = 5

    # 改行を追加する
    text_list = []
    max_text_width = width - 20  # 余白を考慮してテキストの幅を縮める
    while len(text) > 0:
        line = text[:20]  # 1行あたり20文字まで表示する
        if len(line) == 20 and line[-1] not in ['。', '、']:
            # 改行位置を調整する
            idx = line.rfind('。')
            if idx == -1:
                idx = line.rfind('、')
            if idx != -1:
                line = line[:idx+1]
        text_list.append(line.strip())
        text = text[len(line):]

    text = '\n'.join(text_list)

    # text_size = cv2.getTextSize(text, fontFace, font_scale, thickness)[0]
    # text_x = int(width + (width - text_size[0]) / 2)
    # text_y = int((height + text_size[1]) / 2)
    text_x = int(width + (width / 2)-120)
    text_y = int((height) / 2)

    img = cv2_putText_1(white_bg, text, (text_x, text_y), fontFace, font_scale, color)

    # 画像を表示
    # cv2.imshow('Image with Text', img)
    # cv2.waitKey(0)

    # # 画像を保存
    # cv2.imwrite('image_with_text.jpg', white_bg)

    return img

#----------------------
# メイン
#----------------------

if st.button("生成"):
    prog = 0
    my_bar = st.progress(prog)

    st.write('...物語生成中')
    story = generate_story(pages, age)
    print('生成：',story)
    prog=prog+10
    my_bar.progress(prog)
    

    st.write('...物語整形中')
    story = story_line(story)
    print('整形：',story)
    prog=prog+10
    my_bar.progress(prog)
    

    st.write('...画像生成中')
    prompt_list = []
    for i,line in enumerate(story.splitlines()):        
        # st.session_state.j_list = j_list

        line_e = generate_prompt(line)

        if line is None:
            prog=min(100, prog+int(80/len(story.splitlines())))
            my_bar.progress(prog)
            break
        
        if i == 0:
            preb_line = None
        else:
            print(prompt_list[-1])
            print(line)
            preb_line = prompt_list[-1]

        img = generate_t2i(line_e,pages,style,preb_line)
        st.write(line)
        img = with_text(text=line,img=img)
        st.image(img, use_column_width = "auto")


        prog=min(100, prog+int(80/len(story.splitlines())))
        my_bar.progress(prog)


        prompt_list.append(line_e)
        # if i >= 4:
        #     break





# if st.button("絵本作成"):
#     # for i,page in enumerate(st.session_state.e_list):
#     for i,eigo in enumerate(st.session_state.e_list):
#         # st.write(i)
#         img = generate_t2i(eigo,pages)
#         st.image(img, use_column_width = True)

#----------------------
# 編集モード
#----------------------


# from PIL import Image, ImageOps

# st.title("画像編集アプリ")

# # ファイルアップロード
# file = st.file_uploader("画像をアップロードしてください。", type=["jpg", "png"])

# if file is not None:
#     # 画像の読み込み
#     image = Image.open(file)

#     # 画像の表示
#     st.image(image, caption="アップロードされた画像", use_column_width=True)

#     # 画像の編集
#     option = st.selectbox("編集オプションを選択してください。", ("オリジナル", "グレースケール", "反転"))

#     if option == "グレースケール":
#         image = image.convert('LA')
#         st.image(image, caption="グレースケール画像", use_column_width=True)

#     elif option == "反転":
#         image = ImageOps.invert(image)
#         st.image(image, caption="反転画像", use_column_width=True)

#     else:
#         st.image(image, caption="オリジナル画像", use_column_width=True)



# from streamlit_drawable_canvas import st_canvas

# canvas = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",
#     stroke_width=2,
#     stroke_color="rgb(0, 0, 0)",
#     background_color="rgb(255, 255, 255)",
#     height=200,
#     width=200,
#     drawing_mode="freedraw",
#     key="canvas",
# )

# if canvas.image_data is not None:
#     # キャンバス上でのマウスの位置を取得する
#     y, x = np.where(canvas.image_data[:, :, 3] != 0)
#     if len(x) > 0 and len(y) > 0:
#         st.write(f"マウスの位置: ({x[0]}, {y[0]})")


# # Specify canvas parameters in application
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )

# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

# realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# # Create a canvas component
# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,
#     drawing_mode=drawing_mode,
#     point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
#     key="canvas",
# )

# # Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
        
