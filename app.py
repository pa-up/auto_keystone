import streamlit as st
from views import keystone

st.title('画像加工アプリ')
st.write('\n')

def change_page(page):
    st.session_state["page"] = page

def keystone_page():
    # 画像処理の実行
    page_subtitle = "<h3>画像を台形補正</h3><p></p>"
    keystone.main(page_subtitle)


# メイン
def main():
    # セッション状態を取得
    session_state = st.session_state

    # セッション状態によってページを表示
    if "page" not in session_state:
        session_state["page"] = "keystone_page"

    if session_state["page"] == "keystone_page":
        keystone_page()

if __name__ == "__main__":
    main()
