# -*- coding: utf-8 -*-
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
from callbacks import KGCNMetric
from models.base_model import BaseModel
from config import ModelConfig
from models.AttentionMode import GAT,inter_GAT
import tensorflow as tf
from models.SAGPooling import SAGPooling
from models.inter_attention import SelfAttention


class GetReceptiveField(Layer):
    '''
    图处理模块
    对图进行操作
    '''
    
    def __init__(self,config:ModelConfig,name='receptive_filed',**kwargs):
        super(GetReceptiveField,self).__init__(name = name,**kwargs) 
        self.config = config
        
    def call(self,x):
        
        neigh_ent_list = [x]
        neigh_rel_list = []
        n_neighbor = K.shape(self.config.adj_entity)[1]

        for i in range(2):

            new_neigh_ent = K.gather(self.config.adj_entity, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            new_neigh_rel = K.gather(self.config.adj_relation, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))
        return neigh_ent_list + neigh_rel_list


class SqueezeLayer(Layer):
    '''
    降为模块
    '''
    def __init__(self,name='squeeze',**kwargs):
        super(SqueezeLayer,self).__init__(name = name,**kwargs)
    def call(self,x):
        return K.squeeze(x, axis=1)


################### 
#

##################
class SigmoidLayer(Layer):
    '''
    得分预测模块
    '''
    def __init__(self,name='sigmoid',**kwargs):
        super(SigmoidLayer,self).__init__(name = name,**kwargs)
    def call(self,x):
        return K.sigmoid(K.sum(x[0] * x[1], axis=-1, keepdims=True))

class RGDA(BaseModel):
    '''
    模型
    '''
    def __init__(self, config):
        super(RGDA, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            shape=(1, ), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            shape=(1, ), name='input_drug_two', dtype='int64') 
           
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.ent_embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')

        ############对图进行编码
        # Drug one
        get_receptive_field_one = GetReceptiveField(self.config,name='receptive_filed_drug_one')
        #print("input_drug_one:",input_drug_one)
        receptive_list_drug_one = get_receptive_field_one(input_drug_one)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth+1]

        #print("neigh_rel_list_drug_one:",neigh_rel_list_drug_one)
        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]

        # Drug two
        get_receptive_field = GetReceptiveField(self.config,name='receptive_filed_drug')
        receptive_list = get_receptive_field(input_drug_two)
        neigh_ent_list = receptive_list[:self.config.n_depth+1]
        
        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        

        

        e_drug_one = neigh_ent_embed_list_drug_one[0]
        lc_list_one = [] 
        e_drug_two = neigh_ent_embed_list[0]
        lc_list_two = []




        ################GATGAT

        attention_scale_one = GAT(
            self.config,
            self.config.aggregator_type,
            regularizer=l2(self.config.l2_weight),
            name = f'cross_attention_{0}_one')
        inter = inter_GAT(
            self.config,
            self.config.aggregator_type,
            regularizer=l2(self.config.l2_weight*2),
            name=f'cross_attention_{0}_two')

        #####SAGPooling
        pool = SAGPooling()
        hop = 0
        temp_one = neigh_ent_embed_list_drug_one[hop + 1]
        temp = neigh_ent_embed_list[hop + 1]

        def block(drug,adj):
            drug1 = tf.identity(drug)
            drug = attention_scale_one([drug1, adj])
            drug = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                tf.add(drug1, drug))
            drug2 = tf.identity(drug)
            drug3 ,_= pool([drug2, adj])
            drug4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                tf.add(drug2, drug3))
            drug5 = tf.add(drug1,drug4)
            return drug5
        def inter_block(drug,adj):
            drug1 = tf.identity(drug)
            drug = inter([drug1, adj])
            drug = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                tf.add(drug1, drug))
            drug2 = tf.identity(drug)
            drug3 ,_= pool([drug2, adj])
            drug4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                tf.add(drug2, drug3))
            drug5 = tf.add(drug1,drug4)
            return drug5
        def block(drug,adj):
            drug1 = tf.identity(drug)
            drug = attention_scale_one([drug1, adj])
            drug = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                drug)
            drug2 = tf.identity(drug)
            drug3 ,_= pool([drug2, adj])
            drug4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
             drug3)

            return drug4
        def inter_block(drug,adj):
            drug1 = tf.identity(drug)
            drug = inter([drug1, adj])
            drug = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                drug)
            drug2 = tf.identity(drug)
            drug3 ,_= pool([drug2, adj])
            drug4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                drug3)
            return drug4


        #########block
        #drugone
        neigh_ent_embed_list_drug_one1 = tf.identity(neigh_ent_embed_list_drug_one[hop])
        for j in range(self.config.number):
            neigh_ent_embed_list_drug_one[hop] = block(neigh_ent_embed_list_drug_one[hop],temp_one)
        print("1",neigh_ent_embed_list[hop])
        #drugtwo
        neigh_ent_embed_list1 = tf.identity(neigh_ent_embed_list[hop])
        for i in range(self.config.number):
            neigh_ent_embed_list[hop] = block(neigh_ent_embed_list[hop],temp)





        #########interduag_GAT

        drug = K.concatenate([neigh_ent_embed_list_drug_one1,neigh_ent_embed_list1],axis=1)
        print("drug", drug)
        adj = K.concatenate([temp_one,temp],axis=1)
        for i in range(self.config.number):
            print("ff",drug)
            drug = inter_block(drug,adj)
        print("d",drug)

        #########特征融合
        lc_list_one.append(neigh_ent_embed_list_drug_one[0])
        lc_list_two.append(neigh_ent_embed_list[0])
        e_drug_one = K.concatenate([e_drug_one,K.concatenate(lc_list_one)])
        e_drug_two = K.concatenate([e_drug_two,K.concatenate(lc_list_two)])



        #####降维
        squeeze_layer = SqueezeLayer()
        drug1_squeeze_embed = squeeze_layer(e_drug_one)
        drug2_squeeze_embed = squeeze_layer(e_drug_two)


        drug4 = tf.reshape(drug, [-1, 1, drug.shape[1] * drug.shape[2]])

        drug4 = tf.reshape(drug4, [-1, 1, drug4.shape[2]])
        drug4_squeeze_embed = squeeze_layer(drug4)

        print("drug4_emb",drug4_squeeze_embed)


        #####attention层
        att = SelfAttention()
        print("atten_1",drug1_squeeze_embed,drug4_squeeze_embed)
        drug1_squeeze_embed ,_= att([drug1_squeeze_embed,drug4_squeeze_embed])
        drug2_squeeze_embed ,_= att([drug2_squeeze_embed,drug4_squeeze_embed])
        print("atten_2",drug1_squeeze_embed)

        # #### attention---->add
        # print("cat", drug1_squeeze_embed, drug4_squeeze_embed)
        # drug1_squeeze_embed = tf.add(drug1_squeeze_embed,drug4_squeeze_embed)
        # drug2_squeeze_embed = tf.add(drug2_squeeze_embed,drug4_squeeze_embed)
        # print("cat_2", drug1_squeeze_embed)

        ######得分预测层
        sigmoid_layer = SigmoidLayer()
        drug_drug_score = sigmoid_layer([drug1_squeeze_embed, drug2_squeeze_embed])
        print("score ",drug_drug_score)


        model = Model([input_drug_one, input_drug_two], drug_drug_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])
        #model.summary()
        return model
    

    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()
        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def predict_attention(self,x):
        neighbor_model = Model(inputs = self.model.input,outputs = [self.model.get_layer('receptive_filed_drug_one').output,self.model.get_layer('receptive_filed_drug').output])
        at_model = Model(inputs = self.model.input,outputs = [self.model.get_layer('cross_attention_0_one').output[1],self.model.get_layer('cross_attention_0').output[1]])
        
        return neighbor_model(x),at_model.predict(x)

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        return auc, acc, f1, aupr



