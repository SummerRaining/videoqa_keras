"""
Created on Fri Jul 27 20:06:50 2018

@author: tunan
"""
import numpy as np
import json
import h5py
from tqdm import tqdm
import os,sys
from collections import Counter
sys.path.append("..")
from config import Config

from sklearn.utils import shuffle
import enchant

'''
将train文件编码成(img,question,answer1,answer2,answer3)的形式
img:图像的名字
question:19个整数
answer1,2,3: 分别对应一个整数
'''
class pretreat_embedAndText(object):
    def __init__(self):
        self.opt = Config()
        self.error_word = []
        self.d = enchant.Dict("en_US")
        self.seed = 2018
    
    #create data,return a list constitude of ['video name','question','answer1','2','3']
    def create_data(self,train_text_path):
        with open(train_text_path) as f:
            lines = f.readlines()
        question_index = [1,5,9,13,17]
        data = []
        for line in lines:
            Lline = line.strip().split(',')
            img_name = Lline[0]
            for question in question_index:
                answer = [Lline[question+i+1] for i in range(3)]
                data.append([img_name,Lline[question]]+answer)
        return data        
    #encode data. 
    def encode_data(self,data,wordans_to_index_path):
        '''
        input: a list constitude of ['video name','question','answer1','2','3']
        output:  a list consist of  [video_name,encode_question,answer_index1,2,3]
        '''
        if not os.path.exists(wordans_to_index_path):
            raise("no wordans_to_index_path")
        with open(wordans_to_index_path) as f:              
            wordans_to_index = json.load(f)
        word_to_index = wordans_to_index['word_to_index']
        answer_to_index = wordans_to_index['ans_to_index']
        
        encode_data = []
        for sample in data:
            answers = []
            #有答案不在答案表里面，就用其他答案代替，如果都不在就舍弃这个样本
            for answer in sample[2:]:
                if answer in answer_to_index:
                    answers.append(answer_to_index[answer])
            if answers == []:
                continue
            if len(answers)== 2:
                answers.append(answers[0])
            if len(answers)== 1:
                answers.append(answers[0])
                answers.append(answers[0])
            encode_data.append([sample[0],self.convert_question(sample[1],word_to_index)]+answers)
        return encode_data

    def error_correction(self,question,word_to_index):
        '''
         去除's,'符号，对不在glove中的词检查拼写错误纠错，
         对train中的词来讲，由于word_to_index包含了glove和train中所有词的交集，
         故不在word_to_index中的词也一定不在glove中。
        '''
        question = question.replace('\'s','' ).replace('\'','')
        new_question = []
        for x in question.split():
            if x not in word_to_index:
                x = self.d.suggest(x)[0]
            new_question.append(x)
        return " ".join(new_question)
        
    #将一个句子转成19个数字
    def convert_question(self,question,word_to_index,seq_lenth = 19):
        question = self.error_correction(question,word_to_index)
        convert_que = np.zeros(seq_lenth,dtype = np.int32)-1
        i = 0
        for x in question.split():
            if x in word_to_index:
                convert_que[i] = word_to_index[x]
                i += 1
        return convert_que
    
    #将样本编码写入文件中
    def write_encode(self,train_encode,path_name):
        Ltrain_encode = []
        for [name,question,answer1,answer2,answer3] in train_encode:    
            question = list(map(str,question.tolist()))
            Ltrain_encode.append([name]+ question +[str(answer1),str(answer2),str(answer3)])
        text = "\n".join([",".join(line) for line in Ltrain_encode])
        with open(path_name,'w') as f:
            f.write(text)
            
    def text_encode(self):
        '''
        生成编码数据，并分隔成训练集和测试集，写入文件。
        '''
        opt = self.opt
        train_text_path = opt.train_text_path
        wordans_to_index_path = opt.wordans_to_index_path
        train_encode_path = opt.train_encode_path
        val_encode_path = opt.val_encode_path
        
        data = self.create_data(train_text_path)        #产生编码样本
        encode_data = self.encode_data(data,wordans_to_index_path)
        
        #将数据划分，划分train_encode,val_encode
        encode_data = shuffle(encode_data,random_state = self.seed)
        train_encode = encode_data[:int(len(encode_data)*0.9)]
        val_encode = encode_data[int(len(encode_data)*0.9):]
                
        self.write_encode(train_encode,train_encode_path) 
        self.write_encode(val_encode,val_encode_path) 
        
    #generate word embedding matrix
    def build_embed_matrix(self):
        opt = self.opt
        embed_matrix_path = opt.embed_matrix_path
        glove_path = opt.glove_path
        wordans_to_index_path = opt.wordans_to_index_path
        embedding_size = opt.embedding_size
            
        embedding_index = {}
        with open(glove_path,'r',encoding = 'utf-8') as f:     
            #生成word与向量的对应字典
            for line in tqdm(f):
                Lline = line.strip().split()
                word = Lline[0]
                ceof = np.asarray(Lline[1:],dtype = 'float32')
                embedding_index[word] = ceof
                
        #记录词与序号的对应关系        
        with open(wordans_to_index_path) as f:              
            wordans_to_index = json.load(f)
        word_to_index = wordans_to_index['word_to_index']
        
        #矩阵中第i行与第i个词的词向量对应
        embed_matrix = np.zeros([len(word_to_index),embedding_size])   
        for word,index in tqdm(word_to_index.items()):
            if word in embedding_index:
                embed_matrix[index,:] = embedding_index[word]
                
        if os.path.exists(embed_matrix_path):
            print("already exist embed matrix!")
            print("delete origin embed matrix!")
            os.remove(embed_matrix_path)
        #将该文件保存
        with h5py.File(embed_matrix_path) as h:
            h.create_dataset("embed_matrix",data = embed_matrix)
            
    def get_vocabulary(self,glove_path,train_text_path):
        words = []
        #400000个word
        with open(glove_path,'r',encoding = 'utf-8') as f:
            for line in f:
                Lline = line.strip().split()
                words.append(Lline[0])
        words = set(words)
        #提取问题    
        with open(train_text_path) as f:
            train_data = f.readlines()
        questions = []
        question_indexs = [1,5,9,13,17]
        for line in train_data:
            Lline = line.strip().split(',')
            for q in question_indexs:
                questions.append(Lline[q])
                
        #所有问题先检查纠错，去除','s，将不在glove中词汇转化成enchant的词。
        for i in range(len(questions)):
            questions[i] = self.error_correction(questions[i],words)
        q_word = []
        for line in questions:
            q_word += line.split()
        q_word = set(q_word)
        
        #所有问题纠错后的词，与glove的词做交集，就是最后的词汇表
        vocabulary = {w for w in q_word if w in words}
        
        return vocabulary
        
    #答案字典{answer:index}，问题词字典{word:index}
    def save_wordans_index(self):
        opt = self.opt
        glove_path = opt.glove_path
        wordans_to_index_path = opt.wordans_to_index_path
        train_text_path = opt.train_text_path        
        answer_num = opt.answer_num
        #提取所有处理过的词
        vocabulary = self.get_vocabulary(glove_path,train_text_path)
        word_to_index = {x:i for i,x in enumerate(vocabulary)}
        
        #提取答案    
        with open(train_text_path) as f:
            train_data = f.readlines()
        answers = []
        answer_index = [[2,3,4],[6,7,8],[10,11,12],[14,15,16],[18,19,20]]
        for line in train_data:
            '''
            修改这里
            '''
            x = line.split(',')
#            x = line.strip().split(',')
            for index in answer_index:  #所有答案对应的下标列出来。
                answers.append(x[index[0]])
                answers.append(x[index[1]])
                answers.append(x[index[2]])
                
        voc_freq = Counter(answers)     
        #使用Counter中most_common()统计出现的频率,1000类的分类问题，87%
        ans = [word for word,_ in voc_freq.most_common(answer_num)]
        ans_to_index = {x:i for i,x in enumerate(ans)}
        
        wordans_to_index = {'ans_to_index': ans_to_index,'word_to_index': word_to_index}
        with open(wordans_to_index_path,'w') as f:
            json.dump(wordans_to_index,f)
            
def main(**kwargs):
    pretreator = pretreat_embedAndText()
    '''
    word_to_index与embed_matrix对应，所有的模型都与ans_to_index和word_to_index对应。
    修改这两个文件后会使所有训练出的模型不可用。谨慎修改！！！！
    '''
    pretreator.save_wordans_index()    
    pretreator.text_encode()
    pretreator.build_embed_matrix()    #word_to_index重新运行后，embed_matrix必须重新运行
    
if __name__ == '__main__':            
    os.chdir("..")
    import fire
    fire.Fire(main)

    
        