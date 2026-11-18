from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax
from tensorflow.keras import backend as K
from config import ModelConfig


class GAT(Layer):
    def __init__(self,config:ModelConfig,agg: str ,initializer='glorot_normal', regularizer=None,name='GAT',**kwargs):
        super(GAT,self).__init__(name = name,**kwargs)
        self.agg = agg
        self.config = config
        self.initializer = initializer         
        self.regularizer = regularizer
    def build(self, input_shape):
        # 
        self.outputdim = int(self.config.ent_embed_dim/self.config.head)
        self.w0 = self.add_weight(name=self.name+'w0',
                                      shape=(self.config.head,self.config.ent_embed_dim, self.outputdim),
                                      initializer=self.initializer, regularizer=self.regularizer) 

        self.w1 = self.add_weight(name=self.name+'w1',
                                      shape=(self.config.head,2*self.config.ent_embed_dim, self.outputdim),
                                      initializer=self.initializer, regularizer=self.regularizer) 
        self.kernel = self.add_weight(name=self.name+'a',
                                      shape=(self.config.head,self.outputdim+self.outputdim, 1),
                                      initializer=self.initializer, regularizer=self.regularizer)

        #print("+++++++++++",input_shape)
        super(GAT, self).build(input_shape)  
    def call(self,x):
        n_neighbor = self.config.neighbor_sample_size
        n_shape = int(int(x[1].shape[1])/self.config.neighbor_sample_size)
        temp = []
        #
        for i in range(n_shape):
            temp1 = [ x[0][:,i,:] for j in range(n_neighbor)]
            temp1 = K.stack(temp1,1)
            temp.append(temp1)
        x[0] = K.concatenate(temp,1)
        ent = []
        for j in range(self.config.head):
            drug_rel_score = K.relu(K.dot(K.concatenate([K.dot(x[0],self.w0[j]),K.dot(x[1],self.w0[j])]),self.kernel[j]),alpha=0.1)
            
            drug_rel_score = K.reshape(drug_rel_score,
                                        (K.shape(drug_rel_score)[0], n_shape,
                                        n_neighbor, 1))
            attention = softmax(drug_rel_score,2)
            attention = K.reshape(attention,
                                        (K.shape(attention)[0], n_shape*n_neighbor, 1))
            if self.agg == 'neigh':
                weighted_ent = attention * K.dot(x[1],self.w0[j])
            else:
                weighted_ent = attention * K.dot(K.concatenate([x[0],x[1]]),self.w1[j])
            
            weighted_ent = K.reshape(weighted_ent,
                                        (K.shape(weighted_ent)[0], n_shape,
                                        n_neighbor, int(weighted_ent.shape[2])))
            ent.append(K.sum(weighted_ent, axis=2)) 
       
        return K.relu(K.concatenate(ent),alpha = 0.1)


class inter_GAT(Layer):
    def __init__(self, config: ModelConfig, agg: str, initializer='glorot_normal', regularizer=None, name='GAT',
                 **kwargs):
        super(inter_GAT, self).__init__(name=name, **kwargs)
        self.agg = agg
        self.config = config
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        #
        self.outputdim = int(self.config.ent_embed_dim / self.config.head)
        self.w0 = self.add_weight(name=self.name + 'w0',
                                  shape=(self.config.head, self.config.ent_embed_dim, self.outputdim),
                                  initializer=self.initializer, regularizer=self.regularizer)
        #
        self.w1 = self.add_weight(name=self.name + 'w1',
                                  shape=(self.config.head, 2 * self.config.ent_embed_dim, self.outputdim),
                                  initializer=self.initializer, regularizer=self.regularizer)
        self.kernel = self.add_weight(name=self.name + 'a',
                                      shape=(self.config.head, self.outputdim + self.outputdim, 1),
                                      initializer=self.initializer, regularizer=self.regularizer)


    def call(self, x):
        n_neighbor = self.config.neighbor_sample_size#16
        n_shape = int(int(x[1].shape[1]) / (self.config.neighbor_sample_size))
        temp = []
        #
        for i in range(n_shape):
            temp1 = [x[0][:, i, :] for j in range(n_neighbor)]
            temp1 = K.stack(temp1, 1)
            temp.append(temp1)
        x[0] = K.concatenate(temp, 1)
        ent = []
        for j in range(self.config.head):
            drug_rel_score = K.relu(
                K.dot(K.concatenate([K.dot(x[0], self.w0[j]), K.dot(x[1], self.w0[j])]), self.kernel[j]), alpha=0.1)

            drug_rel_score = K.reshape(drug_rel_score,
                                       (K.shape(drug_rel_score)[0], n_shape,
                                        n_neighbor, 1))
            attention = softmax(drug_rel_score, 2)
            attention = K.reshape(attention,
                                  (K.shape(attention)[0], n_shape * n_neighbor, 1))
            if self.agg == 'neigh':
                weighted_ent = attention * K.dot(x[1], self.w0[j])
            else:
                weighted_ent = attention * K.dot(K.concatenate([x[0], x[1]]), self.w1[j])

            weighted_ent = K.reshape(weighted_ent,
                                     (K.shape(weighted_ent)[0], n_shape,
                                      n_neighbor, int(weighted_ent.shape[2])))
            ent.append(K.sum(weighted_ent, axis=2))

        return K.relu(K.concatenate(ent), alpha=0.1)



