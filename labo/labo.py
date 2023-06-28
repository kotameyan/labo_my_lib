# Python 標準ライブラリ
import json
import os
import re
import shutil

# サードパーティのライブラリ
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------------

'''
説明：特定のフォルダ内の画像を一括表示する関数
引数：複数の画像が入ったフォルダのパス
'''
def batch_display(folder_path):

    # 画像ファイルの拡張子（ここでは".jpg"と".png"を対象としています）
    extensions = (".jpg", ".png")

    # 画像ファイルのパスを格納するリスト
    image_paths = []

    # フォルダ内のファイルを走査して画像ファイルのパスをリストに追加する
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extensions):
            image_path = os.path.join(folder_path, file_name)
            image_paths.append(image_path)

    # 画像を順番に読み込んでリストに格納する
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)

    # 画像のサイズを取得
    widths, heights = zip(*(i.size for i in images))

    # 結合後の画像のサイズを計算
    max_width = max(widths)
    total_height = sum(heights)

    # 空のキャンバスを作成
    combined_image = Image.new("RGB", (max_width, total_height))

    # 画像をキャンバスに貼り付ける
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.height

    # 画像を表示
    combined_image.save("/Users/matsumotokotarou/Desktop/combined_image.jpg")

# ------------------------------------------------------------------------------------

'''
説明：各モデルの検証結果（JSON）をval_jsonフォルダに保存する関数
引数：detectまでのパス
'''
def save_val_json(detect_folder_path):
    # 1. detectフォルダ直下にval_jsonフォルダを新たに作成
    val_json_path = os.path.join(detect_folder_path, 'val_json')
    os.makedirs(val_json_path, exist_ok=True)

    # 2. "val"と名前のつく全てのフォルダからpredictions.jsonを取ってきて、val_jsonフォルダに格納
    for folder in os.listdir(detect_folder_path):
        if re.match(r'val.*', folder):
            src_file = os.path.join(detect_folder_path, folder, 'predictions.json')
            if os.path.isfile(src_file):
                # valフォルダの場合は名前をpredictions.json、それ以外はpredictionsX.jsonにする
                dst_file_name = 'predictions.json' if folder == 'val' else f'predictions{folder[3:]}.json'
                dst_file = os.path.join(val_json_path, dst_file_name)
                shutil.copy(src_file, dst_file)

# ------------------------------------------------------------------------------------

'''
説明：val_jsonフォルダ内の全てのJSONファイルを読み込み、グラフを描画する関数
引数：val_jsonフォルダまでのパス
'''
def analyze_val_json(val_json_path):
    # データフレームを保存するための空のリスト
    dataframes = []

    # ディレクトリ内の全てのJSONファイル名を取得
    json_files = [f for f in os.listdir(val_json_path) if f.endswith('.json')]

    # JSONファイル名を数値順に並べ替える
    json_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))

    # 並べ替えたリストの全てのファイルをループ処理
    for filename in json_files:
        print(f"Loading {filename}...")  # 読み込んでいるファイル名を出力
        # ファイルパスを作成
        file_path = os.path.join(val_json_path, filename)
        # JSONファイルを読み込む
        with open(file_path, 'r') as f:
            data = json.load(f)
        # JSONデータをデータフレームに変換
        df = pd.json_normalize(data)
        # 新しい列を作成してグループ名（ファイル名）を格納
        group = re.findall(r'\d+', filename) # 数字を抽出
        df['group'] = group[0] if group else 'unknown' # 数字がなければ'unknown'とする
        # データフレームをリストに追加
        dataframes.append(df)

    # 全てのデータフレームを一つに結合
    result = pd.concat(dataframes, ignore_index=True)

    # scoreをfloat型に変換
    result['score'] = result['score'].astype(float)

    # image_idとgroupでグループ化し、scoreの平均を計算
    grouped = result.groupby(['image_id', 'group'])['score'].mean().reset_index()

    # 結果を確認
    print(grouped)

    # データの可視化
    plt.figure(figsize=(20,10)) # グラフのサイズを設定
    sns.barplot(x='image_id', y='score', hue='group', data=grouped) # 棒グラフの作成

    plt.title('Average Score per Image ID for Each Group') # タイトル
    plt
