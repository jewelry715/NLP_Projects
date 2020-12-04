# NLP_Projects:Attention&Seq2Seq를 활용한 뉴스 헤드라인 요약

## 기획 의도
딥러닝모델로 만든 헤드라인이 원문의 내용을 얼만큼 담고 있는지 확인하고자 한다.
## 데이터 전처리 과정
  1. news_summary,news_summary_more에서 headlines,texts만 추출해서 합침
  2. 데이터 중복제거 및 특수기호,소유격,글자수가 1개인 것은 제거 하지만 Summary에는 불용어 미제거
  3. 전처리 후 공백 null로 전환 후 새로운 데이터셋 제작
## 모델 설계
어텐션과 Seq2Seq를 사용해 모델을 설계하였다.
  - 입력 시퀀스가 길어지면 출력시퀀스의 정확도가 떨어지는 것을 방지해주기 위함
  - RNN의 기울시 손실 문제와 고정된 크기의 벡터에 모든 정보를 압축할 시 정보손실이 발생하는 것이 때문에 이를 보완하기 위해 사용
1. 텍스트 최대 길이는 35, 요약의 최대 길이는 10으로 지정 후 요약문 앞뒤에 sostoken,eostoken을 명명함
2. 텍스트와 요약 데이터를 8:2로 나누어 훈련데이터,레이블 제작
3. 6회 이하의 단어 수 배제 후 정수 인코딩(레이블,훈련 모두)
4. 빈 샘플 제거(eostoken,sostoken만 들어있는 곳 제거) 후 패딩
5. 훈련 데이터 레이블 수  각각 28590,테스트 데이터,레이블 수 각각 7101
6. 단어집합 크기:텍스트는 10000,요약은 6500개
7. 모델 코드
<pre>
<code>
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=128
hidden_size=128
encoder_inputs=Input(shape=(text_max_len,)) #인코더
enc_emb=Embedding(src_vocab,embedding_dim)(encoder_inputs)#임베딩층

encoder_lstm1=LSTM(hidden_size,return_sequences=True,return_state=True,
                  dropout=0.4,recurrent_dropout=0.4) #lstm1
encoder_output1,state_h1,state_c1=encoder_lstm1(enc_emb)
encoder_lstm2=LSTM(hidden_size,return_sequences=True,return_state=True,
                  dropout=0.4,recurrent_dropout=0.4) #lstm2
encoder_output2,state_h2,state_c2=encoder_lstm2(encoder_output1)
encoder_lstm3=LSTM(hidden_size,return_sequences=True,return_state=True,
                  dropout=0.4,recurrent_dropout=0.4) #lstm3
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

decoder_inputs = Input(shape=(None,))

dec_emb_layer = Embedding(tar_vocab, embedding_dim)#임베딩층
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(hidden_size, return_sequences = True, return_state = True, 
                    dropout = 0.4, recurrent_dropout=0.2)#lstm
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])

decoder_softmax_layer = Dense(tar_vocab, activation = 'softmax')#출력층
decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs) 

from attention import AttentionLayer
attn_layer=AttentionLayer(name='attention_layer')#어텐션 층
attn_out,attn_states=attn_layer([encoder_outputs,decoder_outputs])
# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input=Concatenate(axis=-1,name='concat_layer')([decoder_outputs, attn_out])
#디코더의 출력층
decoder_softmax_layer=Dense(tar_vocab,activation='softmax')
decoder_softmax_outputs=decoder_softmax_layer(decoder_concat_input)
#최종 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()
</code>
</pre>
## 참고 문서 및 산출물
- 참고 데이터: https://github.com/sunnysai12345/News_Summary 
- 어텐션 모델:  https://github.com/thushv89/attention_keras/tree/master/src/layers
