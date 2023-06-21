import streamlit as st
import numpy as np
from typing import List , Tuple
from PIL import Image
import cv2


def display_free_size_img(img_np: np.ndarray = np.array([]), img_title:str= "" , img_np_list: List[np.ndarray] = []) -> None:
    """ 画像をユーザーがサイズを自由に調整して表示できる関数 """
    with st.expander("表示画像のサイズを調整"):
        img_size = st.slider("", min_value=10, max_value=600, value=300, step=30)
    st.write("<p></p>", unsafe_allow_html=True)
    st.write(f"<h5>{img_title}</h5>", unsafe_allow_html=True)
    if len(img_np_list) == 0:
        st.image(img_np, width=img_size)
    # numpy画像のリストを入力した場合
    if len(img_np_list) > 0:
        for loop , img_np in enumerate(img_np_list):
            st.markdown(f"{loop + 1}枚目" , unsafe_allow_html=True)
            st.image(img_np, width=img_size)
    st.write("<p><br></p>", unsafe_allow_html=True)


def relative_square_vertex_coordinates_position(vertex_4_coordinates: np.ndarray):
    """ 
    4つの座標がそれぞれ四角形のどの頂点であるかを取得する関数
    Parameters:
        vertex_4_coordinates (np.ndarray) : 4つの頂点の(y,x)座標を格納した(4,2)次元の配列
    Returns:
        vertex_order1_yx (Tuple[int, int]) : 四角形の左上の(y,x)座標
        vertex_order2_yx (Tuple[int, int]) : 四角形の左下の(y,x)座標
        vertex_order3_yx (Tuple[int, int]) : 四角形の右下の(y,x)座標
        vertex_order4_yx (Tuple[int, int]) : 四角形の右上の(y,x)座標
    """
    # x座標が1番小さい座標とその番号を取得
    x_min = vertex_4_coordinates[0][0]
    x_min_number = 0

    for k in range(1, 4, 1):
        if (vertex_4_coordinates[k][0] <= x_min):
            x_min = vertex_4_coordinates[k][0]
            x_min_number = k

    # x座標が2番目に小さいの座標とその番号を取得
    x_pre_min = 10000000
    for k in range(0, 4, 1):
        if (k != x_min_number):
            if (vertex_4_coordinates[k][0] <= x_pre_min):
                x_pre_min = vertex_4_coordinates[k][0]
                x_pre_min_number = k

    #　座標①と座標②を決定
    if ( vertex_4_coordinates[x_min_number][1]    >=    vertex_4_coordinates[x_pre_min_number][1]):
        vertex_order1_x = vertex_4_coordinates[x_pre_min_number][0]
        vertex_order1_y = vertex_4_coordinates[x_pre_min_number][1]
        vertex_order2_x = vertex_4_coordinates[x_min_number][0]
        vertex_order2_y = vertex_4_coordinates[x_min_number][1]
    if ( vertex_4_coordinates[x_min_number][1]    <=    vertex_4_coordinates[x_pre_min_number][1]):
        vertex_order1_x = vertex_4_coordinates[x_min_number][0]
        vertex_order1_y = vertex_4_coordinates[x_min_number][1]
        vertex_order2_x = vertex_4_coordinates[x_pre_min_number][0]
        vertex_order2_y = vertex_4_coordinates[x_pre_min_number][1]

    # 座標③と座標④の候補を取得
    x_number3 = -1
    for k in range(0, 4, 1):
        if (   (k != x_min_number)   and   (k != x_pre_min_number)   and    (x_number3 == -1)   ):
            x_number3 = k
        if (   (k != x_min_number)   and   (k != x_pre_min_number)   and    (x_number3 != -1)   ):
            x_number4 = k

    #　座標③と座標④を決定
    if (vertex_4_coordinates[x_number3][1] >= vertex_4_coordinates[x_number4][1]):
        vertex_order3_x = vertex_4_coordinates[x_number3][0]
        vertex_order3_y = vertex_4_coordinates[x_number3][1]
        vertex_order4_x = vertex_4_coordinates[x_number4][0]
        vertex_order4_y = vertex_4_coordinates[x_number4][1]
    if ( vertex_4_coordinates[x_number3][1]    <=    vertex_4_coordinates[x_number4][1]):
        vertex_order3_x = vertex_4_coordinates[x_number4][0]
        vertex_order3_y = vertex_4_coordinates[x_number4][1]
        vertex_order4_x = vertex_4_coordinates[x_number3][0]
        vertex_order4_y = vertex_4_coordinates[x_number3][1]
    
    vertex_order1_yx = tuple([vertex_order1_y, vertex_order1_x])
    vertex_order2_yx = tuple([vertex_order2_y, vertex_order2_x])
    vertex_order3_yx = tuple([vertex_order3_y, vertex_order3_x])
    vertex_order4_yx = tuple([vertex_order4_y, vertex_order4_x])

    return vertex_order1_yx , vertex_order2_yx , vertex_order3_yx , vertex_order4_yx


def affine_transformation_by_vertex_coordinates(
        input_img: np.ndarray , row: int , col: int ,
        vertex_order1_yx: Tuple[int, int] , vertex_order2_yx: Tuple[int, int] , 
        vertex_order3_yx: Tuple[int, int] , vertex_order4_yx: Tuple[int, int] ,
    ):
    """ 
    補正前の4つの頂点座標を用いて、アフィン変換する関数
    Parameters:
        input_img (np.ndarray) : 入力画像
        vertex_order1_yx : 四角形の左上の(y,x)座標
        vertex_order2_yx : 四角形の左下の(y,x)座標
        vertex_order3_yx : 四角形の右下の(y,x)座標
        vertex_order4_yx : 四角形の右上の(y,x)座標
    Returns:
        affine_transformed_square (np.ndarray) : アフィン変換された画像
    """
    # 補正後の長方形の辺の長さを計算
    width = vertex_order4_yx[1] - vertex_order1_yx[1]
    height = vertex_order2_yx[0] - vertex_order1_yx[0]

    # 変換に利用する座標
    before_square_dots = np.array([ [vertex_order1_yx[1] , vertex_order1_yx[0]] , [vertex_order2_yx[1], vertex_order2_yx[0]], [vertex_order3_yx[1], vertex_order3_yx[0]], [vertex_order4_yx[1], vertex_order4_yx[0]] ], dtype=np.float32)
    square_dots = np.array([[vertex_order1_yx[1] , vertex_order1_yx[0]], [vertex_order1_yx[1], vertex_order1_yx[0] + height], [vertex_order1_yx[1] + width , vertex_order1_yx[0] + height], [vertex_order1_yx[1] + width , vertex_order1_yx[0]]], dtype=np.float32)
    # 変換行列
    trans_array = cv2.getPerspectiveTransform(before_square_dots, square_dots)
    # 射影変換
    affine_transformed_square = cv2.warpPerspective(input_img, trans_array, (col, row))
    return affine_transformed_square


def get_square_contours(contours:np.ndarray):
    """
    輪郭群から四角形の輪郭のみを抽出する関数
    Parameters:
        contours (np.ndarray) : 輪郭の座標を格納する配列
            contours = cv2.findContours(img_mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Returns:
        edge_4_easy (np.ndarray) : 頂点の数が4つ（四角形）の輪郭
        square_contours_count (int) : 四角形の輪郭の数
    """
    # 輪郭の頂点の数を減らす
    edge_easy = np.empty( (len(contours)) ,  dtype = 'object' )  # 近似輪郭の配列
    edge_4_easy = []

    square_contours_count = 0   # 頂点数が4の輪郭をカウント （0個の場合は、補正処理を禁止＆エラー文言を表示）
    for k in range(0, len(contours), 1):
        # 点の数を減らして輪郭を近似
        edge_long = cv2.arcLength(contours[k], True)
        ratio = 0.02
        edge_easy[k] = cv2.approxPolyDP(contours[k], epsilon=ratio * edge_long, closed=True)

        # 点の数が4つの輪郭の配列のみを保存
        if (len(edge_easy[k]) == 4):
            edge_4_easy.append(edge_easy[k])
            square_contours_count = square_contours_count + 1
    return edge_4_easy , square_contours_count


def auto_keystone(rgb_img_np: np.ndarray , mask_df: int = 20):
    """ 
    台形補正を実行する関数 
        入力画像を複数のmin閾値ごとに2値化し、各々に台形補正画像を出力するs
     Parameters:
        rgb_img_np (np.ndarray) : RGB画像
        mask_df (int) : 複数設定する「2値化のmin閾値」どうしの差
            「mask_df」を小さくする → 精度 & 処理負荷 の増大
    """
    row = rgb_img_np.shape[0]  # 入力画像の行数（縦サイズ）
    col = rgb_img_np.shape[1]   # 入力画像の列数（横サイズ）
    img_gray = cv2.cvtColor(rgb_img_np, cv2.COLOR_BGR2GRAY)  #グレースケール化

    # マスク画像のmin閾値ループに必要な変数と準備
    mask_number = int(250 / mask_df) + 1   # マスク画像の数(min閾値の数)
    img_mask = np.empty((mask_number),  dtype='object')
    available_mask_count = 0   #「点の数が4つの近似輪郭」を含む マスク画像の数
    edge_4_mask = []    # 各マスク画像に対して検出された、面積が最大な「点の数が4つの近似輪郭」の「座標群」のリスト

    # min閾値ごとに条件分岐（マスク画像で頂点4つの画像が得られなかったら、入力画像を入れる）
    for m in range( 0, mask_number ):
        # 2値化処理
        mask_min  =  10  +  m * mask_df   # 2値化のmin閾値（例. 1回：10 , 2回：30 , ... , 13回：250）
        ret, img_mask = cv2.threshold(img_gray, mask_min, 255, cv2.THRESH_BINARY)  # 2値化処理を実行

        # 輪郭を検出し、輪郭の座標を取得
        contours, hierarchy = cv2.findContours(img_mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 小さすぎる輪郭（ノイズなど）と 大きすぎる輪郭（周辺減光や枠線など）を削除
        contours = list(filter(lambda x: cv2.contourArea(x) > (40*40/100/100 * row * col), contours))
        contours = list(filter(lambda x: cv2.contourArea(x) < (99*99/100/100 * row * col), contours))

        # 頂点が4つ（四角形）の輪郭のみを抽出
        edge_4_easy , square_contours_count = get_square_contours(contours)

        # 1つのマスク画像内で、面積が最大の「頂点が4つの輪郭」を リスト edge_4_mask に格納
        if square_contours_count >= 1:
            edge_4_mask.append(max(edge_4_easy, key=lambda x: cv2.contourArea(x)))
            available_mask_count = available_mask_count + 1

    # マスク画像の中で四角形の輪郭が存在する場合のみ、台形補正を実行
    output_img_list = []
    if available_mask_count >= 1:
        for p in range( 0 , len(edge_4_mask) ):
            edge_square = np.squeeze(edge_4_mask[p])  # (4,1,2) → (4,2) に補正
            
            # 4つの座標がそれぞれ四角形のどの頂点であるかを取得
            vertex_order1_yx , vertex_order2_yx , vertex_order3_yx , vertex_order4_yx = \
                relative_square_vertex_coordinates_position(edge_square)
            
            # 4つの頂点座標から長方形に補正
            affine_transformed_square = affine_transformation_by_vertex_coordinates(
                rgb_img_np , row , col  ,
                vertex_order1_yx , vertex_order2_yx , 
                vertex_order3_yx , vertex_order4_yx ,
            )
            output_img_list.append(affine_transformed_square)
    return output_img_list, available_mask_count


def main(page_subtitle="<h1>画像を台形補正</h1><p></p>"):
    # タイトル
    st.write(page_subtitle, unsafe_allow_html=True)

    st.write("<h5>画像をアップロード or ドロップしてください</h5>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("画像は横サイズ、縦サイズともに1500px未満でアップロードしてください。（NG例 : 1800 × 300）", type=["jpg", "jpeg", "png"])
    uploaded_explanation = """
    <span style='font-size:15px;'><ul>
    <li>本サイトでは入力画像が台形でなくても、四角形であれば補正できます。</li>
    <li>「補正したい四角形領域の内部」と「背景」の明度差が大きいほど綺麗に補正されます。</li>
    <li>有効な画像例：<br>「領域内部」：明るい色（or 暗い色） ,「背景」：暗い色（or 明るい色）など</li>
    </ul></span>
    """
    st.markdown(uploaded_explanation , unsafe_allow_html=True)

    if uploaded_file is not None:
        # PILで画像を開く
        img_pil = Image.open(uploaded_file)
        # Numpyの配列に変換
        rgb_img_np = np.array(img_pil)
        # 透明度チャンネルがある場合はRGBに変換
        rgb_img_np = rgb_img_np[..., :3]

        # 台形補正を実行
        cv_calc_img, success_count = auto_keystone(rgb_img_np)

        if success_count == 0:
            st.markdown(f"<h5>台形補正失敗</h5>" , unsafe_allow_html=True)
            st.markdown(f"申し訳ございません！サービスの品質向上に向けて精進いたします！")

        else:
            st.markdown(f"<p><br></p>" , unsafe_allow_html=True)
            st.markdown(f"{success_count}種類の補正画像が得られました。<br>最も綺麗に補正できている画像をお選びください。" , unsafe_allow_html=True)
            st.markdown(f"<p><br></p>" , unsafe_allow_html=True)
            st.markdown(f"<h5>台形補正後の画像</h5>" , unsafe_allow_html=True)
            
            # 補正後の画像を表示
            display_free_size_img(img_np_list=cv_calc_img)
            
            # 補正後の画像をフォルダに保存
            for loop , cv_img in enumerate(cv_calc_img):
                output_img_path = "media/img/" + "ks_img" + str(loop) + ".png"
                cv2.imwrite(output_img_path, cv_img)


    st.markdown(f"<p><br></p>" , unsafe_allow_html=True)
    st.markdown(f"<h5>台形補正とは</h5>" , unsafe_allow_html=True)
    st.markdown(f"長方形の対象物を斜め方向から撮影すると、対象物が台形状に歪んでいる写真が出来上がります。" , unsafe_allow_html=True)
    sample_img_path ="static/img/ks_img1.png"
    sample_img = cv2.cvtColor(cv2.imread(sample_img_path), cv2.COLOR_BGR2RGB)
    st.image(sample_img)
    st.markdown(f"""
        台形補正はそのような台形状に歪んだ画像が綺麗な長方形になるよう補正することができます。
        <br>※ 本サイトでは入力画像が台形でなくても、四角形であれば補正できます！
        """ , unsafe_allow_html=True)

if __name__ == '__main__':
    main()
