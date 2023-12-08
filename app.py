from scipy.sparse import load_npz
import pandas as pd
import pymysql
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from kiwipiepy import Kiwi
import urllib.parse
from flask import Flask, request, jsonify
import re
import pickle
import time

try:
    print("IBCF 불러오기 시작")

    book_ibcf_df = pd.read_csv("/home/Flask_Searver/similarity_rate_df.csv")

    book_ibcf_df.set_index('book_id', inplace=True)

    print("IBCF 불러오기 성공")
except:
    print("IBCF 불러오기 실패")


# DB 접속용
config = {
    'user': 'test-user',
    'password': '1234qwer',
    'host': '49.50.167.140',
    'port': 3306,
    'database': 'testdb',
}

def getTime():
    return time.time()

app = Flask(__name__)

kiwi = Kiwi()

model_file_path = '/home/Flask_Searver/tfidf_vectorizer_model.pkl'
matrix_file_path = "/home/Flask_Searver/document_matrix.npz"

ibcf_file_path = '/home/Flask_Searver/model_item_based.pkl'
train_file_path = "/home/Flask_Searver/trainset.pkl"



# IBCF 모델 불러오기
with open(ibcf_file_path, 'rb') as model_file:
    loaded_ibcf = pickle.load(model_file)

# IBCF 모델 불러오기
with open(train_file_path, 'rb') as model_file:
    loaded_trainset = pickle.load(model_file)


# CBF 모델 불러오기
with open(model_file_path, 'rb') as model_file:
    loaded_vectorizer = pickle.load(model_file)

# CBF 행렬 불러오기
loaded_document_matrix = load_npz(matrix_file_path)


# 문장에서 불필요한 부분 제거
def preprocess_sentence(sentence, name=''):
    if not isinstance(sentence, str):
        print('문자열이 아닙니다.')
        return sentence
    name_parts = name.split(' ')
    if len(name_parts) > 1:
        sentence = re.sub(name_parts[0], '', sentence)
        sentence = re.sub(name_parts[1], '', sentence)

    sentence = re.sub('[^가-힣a-zA-Z0-9]+|[ㄱ-ㅎㅏ-ㅣ]', ' ', sentence)

    return sentence

# 기본 쿼리
def getBookId():
    query = """
        SELECT DISTINCT b.book_id
        FROM book b
    """
    return query

def getKeyword():
    query = """
        SELECT * FROM book_keyword
    """
    return query

def getRating():
    query = """
        SELECT r.user_id, r.book_id, r.rating
        FROM book_rating r
    
    """
    return query

def getHistory():
    query = """
        SELECT * FROM book_history
    """
    return query

def getQuery(query, cursor, param=None):
    cursor.execute(query, param)
    return cursor.fetchall()

##### DB

try:
    conn = pymysql.connect(**config)

    cursor = conn.cursor()

    print('★☆☆☆ connect 성공')

    # 책 ID 조회 쿼리
    query_book_id = getBookId()
    result_book_id = [i[0] for i in getQuery(query_book_id, cursor)]
    # print("책 ID 결과:", result_book_id[:10])

    print('★★☆☆ 쿼리 성공')

    # 키워드 조회 쿼리
    query_keyword = getKeyword()
    keywords = getQuery(query_keyword, cursor)
    # print("키워드 결과:", keywords[:1])

    # 레이팅 조회 쿼리
    query_rating = getRating()
    ratings = getQuery(query_rating, cursor)
    # print("레이팅 결과:", ratings[:1])

    query_history = getHistory()
    historys = getQuery(query_history, cursor)

    print("★★★☆ 쿼리 결과 성공")

except:
    print("☆☆☆☆ mysql db 연결 실패")
try:
    book_id_df = pd.DataFrame({'book_id': result_book_id})

    print("book_id_df : ",book_id_df.columns)
    print()
    
    keyword_df = pd.DataFrame(keywords, columns=["book_id", "keywords"])

    print("keyword_df : ", keyword_df.columns)
    print()
    
    rating_df = pd.DataFrame(ratings, columns=["user_id", "book_id", "rating"])

    rating_df["rating"] += 1

    print("rating_df : ",rating_df.columns)
    print()
    
    history_df = pd.DataFrame(historys, columns=["user_id", "book_id"])

    print("history_df : ",history_df.columns)
    print()

except:
    print("☆☆☆☆ 데이터 프레임 화 실패")
else:
    print("★★★★ 데이터 프레임 화 성공")
##### DB

# 추천 함수 정의
def get_recommendations(cosine_sim, df):
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print("sim_scores : ", sim_scores[:10])
    sim_scores = [score for score in sim_scores if score[1] >= 0.3]
    sim_scores = sim_scores[:10]
    # print("sim_scores : ", sim_scores)
    book_indices = [i[0] for i in sim_scores]
    selected_book_ids = df[df['book_id'].isin(book_indices)]['book_id'].to_list()

    return selected_book_ids

# 사용자 독서 기록을 기반으로 추천하는 함수
def recommend_books_based_on_reading_list(user_reading_df, books_df, tfidf_vectorizer, tfidf_matrix):
    user_keywords = ', '.join(books_df[books_df['book_id'].isin(user_reading_df['book_id'])]['keywords'])
    user_vector = tfidf_vectorizer.transform([user_keywords])
    cosine_sim = linear_kernel(user_vector, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    recommended_books = [books_df['book_id'].iloc[i] for i in sim_scores]
    return recommended_books

# 두 추천 결과를 가중 평균하는 함수
def combine_recommendations(recommendations1, recommendations2, weight1=0.5, weight2=0.5):
    combined_recommendations = []
    for rec1, rec2 in zip(recommendations1, recommendations2):
        combined_rec = int(weight1 * rec1 + weight2 * rec2)
        combined_recommendations.append(combined_rec)
    return combined_recommendations


##### CBF
@app.route('/api/list', methods=['GET'])
def list_books():

    start = time.time()

    keyword = urllib.parse.unquote(request.args.get('keyword'))
    # username = urllib.parse.unquote(request.args.get('username'))

    # print("----->>>>",username)

    ##### CBF

    preprocessed_keyword = preprocess_sentence(keyword)

    print("Keyword : ", keyword)

    if preprocessed_keyword:
        cbf_start = time.time()

        s1 = getTime()
        search_tokens = kiwi.analyze(preprocessed_keyword)[0][0]
        search_keywords = [token[0] for token in search_tokens]
        e1 = getTime()

        s2 = getTime()
        search_vector = loaded_vectorizer.transform([' '.join(search_keywords)])
        print("◎◎◎◎◎ 백터화 성공")
        e2 = getTime()


        s3 = getTime()
        cosine_sim = cosine_similarity(search_vector, loaded_document_matrix)
        print("◎◎◎◎◎ 코사인 성공")
        e3 = getTime()

        s4 = getTime()
        result = get_recommendations(cosine_sim, book_id_df)

        # ### CBF 독서 이력
        # recom_bookRead = recommend_books_based_on_reading_list(history_df, keyword_df, loaded_vectorizer, loaded_document_matrix)
        # print("Book_Base_Recom", recom_bookRead)

        print("◎◎◎◎◎ 추천 성공")
        e4 = getTime()

        end = time.time()
        cbf_end = time.time()

        print("토큰화 걸린 시간 : ", e1 - s1)
        print("백터화 걸린 시간 : ", e2 - s2)
        print("코사인 걸린 시간 : ", e3 - s3)
        print("추천   걸린 시간 : ", e4 - s4)
        print("CBF   걸린  시간 : ", cbf_end - cbf_start)

        print("       걸린 시간 : ", end - start)

        print("result : " ,result)

        return jsonify({'result': result})

        # [2061,2064,2074,2097,9944,10176,10415,10435,10540,10557] 파이썬
        # IBCF :  {1, 2, 3, 4, 6, 8, 9, 10, 11, 16, 17, 18, 24, 28, 31, 33, 35, 39, 40, 44, 45, 46, 51, 54}
    ##### CBF



    response_data = {'message': 'Hello from Flask! This is the response from /list endpoint.'}
    return response_data

@app.route('/api/ibcf', methods=['GET'])
def ibcf_books():

    bookid = int(urllib.parse.unquote(request.args.get('bookid')))

    ibcf = book_ibcf_df[str(bookid)].sort_values(ascending=False)[1:101]
    ibcf = ibcf[ibcf >= 0.1]

    if ibcf:
        return jsonify({'ibcf': ibcf})


    response_data = {'message': 'Hello from Flask! This is the response from /list endpoint.'}
    return response_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
