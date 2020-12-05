import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
import time
from random import shuffle




with open("/Users/thibautgold/Documents/Number_Recognition/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

with open("/Users/thibautgold/Documents/Number_Recognition/synweights97.64%.pkl", "br") as fh:
    synweights = pickle.load(fh)

lr = np.arange(10)
hot_list=[]
for label in range(10):
    one_hot = (lr==label).astype(np.int)
    hot_list.append(one_hot)
 
    
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]

#for i in range(20): 
 #   img = train_imgs[i].reshape((28,28))
 #   plt.imshow(img, cmap="Greys")
#    plt.show()
    

w1=synweights[0]
w2=synweights[1]
w3=synweights[2]

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
neurons_first_layer=image_pixels

big_cost_list=[]

percentage_completed=[]
for i in range(101):
    percentage_completed.append(i/100)

# for i in range(100):
#     img = train_imgs[i].reshape((28,28))
#     plt.imshow(img, cmap="Greys")
#     plt.show()
#     print(train_labels[i])
    
#matrix(row,column)

#input_vector=train_imgs[1]

def trymynumbers():
    
    test_image_1=cv.imread("/Users/thibautgold/Documents/Number_Recognition/numbers_antoine.jpg",cv.IMREAD_GRAYSCALE)
   # plt.imshow(test_image_1)
    #plt.show()
    BLURED= cv.blur(test_image_1,(3,3),1000)
    #plt.imshow(BLURED)
    img1=cv.threshold(BLURED,120,255,cv.THRESH_BINARY)[1]
    plt.imshow(img1)
    plt.show()
   # img=img1.astype(np.uint8)
    img1= np.pad(img1, pad_width=1000, mode='constant', constant_values=1)    

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img1), connectivity=8)
    train_imgs=list()

    for i in range(2, nb_components-1):
        if stats[i][-1]>=5000:
            
            x1=int(centroids[i][0]-stats[i,3]//1.8)
            x2=int(centroids[i][0]+stats[i,3]//1.8)
            y1=int(centroids[i][1]-stats[i,3]//2)
            y2=int(centroids[i][1]+stats[i,3]//2)
            
           # x1=stats[i,0]-50
           # x2=stats[i,0]+stats[i,3]//2+50
            
           # y1=stats[i,1]
           # y2=stats[i,1]+stats[i,3]
          

            img1=np.zeros(np.shape(output))
            img1[output==i]=1
            img2=cv.resize((img1[y1:y2, x1:x2]),(20,20))
            img3= np.pad(img2, pad_width=4, mode='constant', constant_values=0)    

           # img=img.astype(np.uint8)

            BLURED= cv.GaussianBlur(img3,(3,3),0)
            train_imgs.append(BLURED.reshape(784,))
           # plt.imshow(BLURED,cmap="Greys")
           # plt.show()
    return(train_imgs)


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result


def expand_data_set():    
    global train_imgs,train_labels, train_imgs2
    train_imgs2=[]
    train_imgs3=[]
    train_labels3=[]

    for i in range(len(train_imgs)):

        train_imgs2.append((train_imgs[i],train_labels[i])) 
        
        train_imgs2.append((cv.GaussianBlur(train_imgs[i],(3,3),0).reshape(784,),train_labels[i]))
        
        train_imgs2.append((rotate_image(train_imgs[i].reshape(28,28),10).reshape(784,),train_labels[i]))     
        
        train_imgs2.append((rotate_image(train_imgs[i].reshape(28,28),-10).reshape(784,),train_labels[i]))
   
    shuffle(train_imgs2)
   
    for img_label in train_imgs2:
        train_imgs3.append(img_label[0])   
        train_labels3.append(img_label[1])

    train_imgs=np.asarray(train_imgs3)
    train_labels=np.asarray(train_labels3)

    
#expand_data_set()
#print("Data expanded from",len(data[0]),"to",len(train_imgs),"images")



def sigmoid(x):
    s=1/(1+np.exp(-x))
    return(s)

def dsigmoid(x):
    s=x*(1-x)
    return(s)

def logit(x):
    s=np.log((x/(1-x)))
    return(s)
    
def save(best_weights):
    with open("/Users/thibautgold/Documents/Number_Recognition/synweights"+str(round(best_weights[3],3))+"%.pkl", "bw") as fh:
        data =(best_weights[0],best_weights[1],best_weights[2])
        pickle.dump(data, fh)

class NeuralNetwork: 
    def __init__(self,image_pixels,nb_neuron_layer1,nb_neuron_layer2):
        self.nb_neurons_a1=image_pixels
        self.nb_neurons_a2=nb_neuron_layer1
        self.nb_neurons_a3=nb_neuron_layer2
        self.nb_neurons_a4=10
        self.success_rate=0
        self.tested=0
        self.success=0
        self.cost_list=[]
        self.cost_average=0
        self.best_performance=0
        self.success_rate_list=[]

        self.initialize()      
       # self.phoenix()
        #self.birth()


    def initialize(self):
        self.w1=2*np.random.rand(self.nb_neurons_a2,self.nb_neurons_a1)-1
        self.b1=np.zeros(self.nb_neurons_a2)
        
        self.w2=2*np.random.rand(self.nb_neurons_a3,self.nb_neurons_a2)-1
        self.b2=np.zeros(self.nb_neurons_a3)
        
        self.w3=2*np.random.rand(self.nb_neurons_a4,self.nb_neurons_a3)-1
        self.b3=np.zeros(self.nb_neurons_a4)
        
    def phoenix(self):
        self.nb_neurons_a2=np.shape(w1)[0]
        self.nb_neurons_a3=np.shape(w2)[0]

        self.w1=w1.astype(np.float128)
        self.w2=w2.astype(np.float128)
        self.w3=w3.astype(np.float128)
        
        self.b1=np.zeros(self.nb_neurons_a2).astype(np.float128)
        self.b2=np.zeros(self.nb_neurons_a3).astype(np.float128)
        self.b3=np.zeros(self.nb_neurons_a4).astype(np.float128)
    

    def birth(self):
        self.w1=np.ones((self.nb_neurons_a2,self.nb_neurons_a1))
        self.b1=np.zeros(self.nb_neurons_a2)
        
        self.w2=np.ones((self.nb_neurons_a3,self.nb_neurons_a2))
        self.b2=np.zeros(self.nb_neurons_a3)
        
        self.w3=np.ones((self.nb_neurons_a4,self.nb_neurons_a3))
        self.b3=np.zeros(self.nb_neurons_a4)
     
    #def mindreader(self,number):
        #tried and failed, not a square matrix so more unknowns than equations, 
        # still possible to find a solution but will it be meaningful ? 

        
    def run(self,input_vector,train_label,iterations,mutation_rate,ignore_correct_ans=False):
        self.sample_size=1
        self.complete_cost_matrix=np.zeros((10,self.sample_size))
        self.counter=0
        
        for i in range(iterations):
            
            self.a1=input_vector[i]
            self.z2=np.matmul(self.w1,self.a1)-self.b1
            self.a2=sigmoid(self.z2)
            
            self.z3=np.matmul(self.w2,self.a2)-self.b2
            self.a3=sigmoid(self.z3)
            
            self.z4=np.matmul(self.w3,self.a3)-self.b3
            self.a4=sigmoid(self.z4)
            
             
            if ignore_correct_ans=='repeat until correct':
                while  np.argmax(self.a4)!=int(train_label[i]):
                    
                    self.error_a4=((hot_list[int(train_label[i])]-self.a4)**1)/1         
                    self.delta_a4=self.error_a4*dsigmoid(self.a4)
                    
                    self.error_a3=np.matmul(self.delta_a4,self.w3)
                    self.delta_a3=self.error_a3*dsigmoid(self.a3)
                    
                    self.error_a2=np.matmul(self.delta_a3,self.w2)
                    self.delta_a2=self.error_a2*dsigmoid(self.a2)
                    
               
        
                    self.w3+=self.a3*(self.delta_a4.reshape(10,1))*mutation_rate
                    self.w2+=self.a2*(self.delta_a3.reshape(self.nb_neurons_a3,1))*mutation_rate
                    self.w1+=self.a1*(self.delta_a2.reshape(self.nb_neurons_a2,1))*mutation_rate
                
                    self.z2=np.matmul(self.w1,self.a1)-self.b1
                    self.a2=sigmoid(self.z2)
            
                    self.z3=np.matmul(self.w2,self.a2)-self.b2
                    self.a3=sigmoid(self.z3)
                
                    self.z4=np.matmul(self.w3,self.a3)-self.b3
                    self.a4=sigmoid(self.z4)
                    #       self.complete_cost_matrix=np.zeros((10,self.sample_size))
        
            
            
            
            
            
            
            if ignore_correct_ans=='ignore correct':
                if  np.argmax(self.a4)!=int(train_label[i]):
                    
                    self.error_a4=((hot_list[int(train_label[i])]-self.a4)**1)/1         
                    self.delta_a4=self.error_a4*dsigmoid(self.a4)
                    
                    self.error_a3=np.matmul(self.delta_a4,self.w3)
                    self.delta_a3=self.error_a3*dsigmoid(self.a3)
                    
                    self.error_a2=np.matmul(self.delta_a3,self.w2)
                    self.delta_a2=self.error_a2*dsigmoid(self.a2)
                    
               
        
                    self.w3+=self.a3*(self.delta_a4.reshape(10,1))*mutation_rate
                    self.w2+=self.a2*(self.delta_a3.reshape(self.nb_neurons_a3,1))*mutation_rate
                    self.w1+=self.a1*(self.delta_a2.reshape(self.nb_neurons_a2,1))*mutation_rate
                
            #       self.complete_cost_matrix=np.zeros((10,self.sample_size))
                   
            if ignore_correct_ans=='every image':             
                self.error_a4=((hot_list[int(train_label[i])]-self.a4)**1)/1         
                self.delta_a4=self.error_a4*dsigmoid(self.a4)
                
                self.error_a3=np.matmul(self.delta_a4,self.w3)
                self.delta_a3=self.error_a3*dsigmoid(self.a3)
                
                self.error_a2=np.matmul(self.delta_a3,self.w2)
                self.delta_a2=self.error_a2*dsigmoid(self.a2)
                
           
    
                self.w3+=self.a3*(self.delta_a4.reshape(10,1))*mutation_rate
                self.w2+=self.a2*(self.delta_a3.reshape(self.nb_neurons_a3,1))*mutation_rate
                self.w1+=self.a1*(self.delta_a2.reshape(self.nb_neurons_a2,1))*mutation_rate
            
        #       self.complete_cost_matrix=np.zeros((10,self.sample_size))
               
    
    def test_performance(self,input_vector,train_label,iterations):
        self.success=0
        for i in range(iterations):
            self.a1=input_vector[i]
            self.z2=np.matmul(self.w1,self.a1)-self.b1
            self.a2=sigmoid(self.z2)
            
            self.z3=np.matmul(self.w2,self.a2)-self.b2
            self.a3=sigmoid(self.z3)
            
            self.z4=np.matmul(self.w3,self.a3)-self.b3
            self.a4=sigmoid(self.z4)
        
          #  print(self.a4)
         #   print("Guess:", np.argmax(self.a4),"Answer:",int(train_label[i]))
          #  img = input_vector[i].reshape((28,28))
           # plt.imshow(img, cmap="Greys")
          #  plt.show()
          #  print("Guess:",np.argmax(self.a4),"Answer:",train_label[i])
            
            if  np.argmax(self.a4)==int(train_label[i]):
                self.success+=1
          #  else:
          #      print(self.a4)
            #    print(np.argmax(self.a4),int(train_label[i]))
        if (self.success/iterations)*100>=self.best_performance:
            self.best_performance=(self.success/iterations)*100
            self.best_wheights=(self.w1,self.w2,self.w3,self.success/iterations*100)

            
        print("Current Success Rate:",(self.success/iterations)*100,"%")
        self.success_rate_list.append((self.success/iterations*100))
        
    def test_my_numbers(self):
        input_vector=trymynumbers()
        for i in range(len(input_vector)):
            self.a1=input_vector[i]
            self.z2=np.matmul(self.w1,self.a1)-self.b1
            self.a2=sigmoid(self.z2)
            
            self.z3=np.matmul(self.w2,self.a2)-self.b2
            self.a3=sigmoid(self.z3)
            
            self.z4=np.matmul(self.w3,self.a3)-self.b3
            self.a4=sigmoid(self.z4)
        
          #  print(self.a4)
         
            img = input_vector[i].reshape((28,28))
            plt.imshow(img, cmap="Greys")
            plt.show()
            print("Guess:", np.argmax(self.a4),"confidence:",np.max(self.a4)*100,"%")

            

        
            
    
BRAIN=[]

#BRAIN.append((NeuralNetwork(image_pixels,100,50),'repeat until correct'))  
BRAIN.append((NeuralNetwork(image_pixels,100,50),'ignore correct'))  
#BRAIN.append((NeuralNetwork(image_pixels,100,50),'every image')  )

#BRAIN[0].test_my_numbers()



Number_of_Cycles=100
Number_of_Images_Per_Cycle=len(train_imgs)
Mutation_Rate=1


for i in range(Number_of_Cycles):
    if i/Number_of_Cycles in percentage_completed:
    
        print("Processed:",int(i/Number_of_Cycles*100),"%")
        for ntw in BRAIN:
            try:      
                ntw[0].test_performance(test_imgs,test_labels,10000)
            except: print("Error")
 
    for ntw in BRAIN:
        start = time.time()
        ntw[0].run(train_imgs,train_labels,Number_of_Images_Per_Cycle,Mutation_Rate,ntw[1])
        end = time.time()
        print("Time taken:",round(end - start,3),"seconds    (",ntw[1],")")





# for ntw in BRAIN:
#     save(ntw.best_wheights)

# for ntw in BRAIN:
#     plt.plot(ntw.success_rate_list)

# plt.show()

liste=[]
for i in range(np.shape(BRAIN[0].w3)[0]):
    for j in range(np.shape(BRAIN[0].w2)[1]):   
        liste.append(BRAIN[0].w2[i,j])
        
for e in liste:
    if e<=-0.065 and e>=-0.14:
        print(e)
        





