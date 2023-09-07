import torch

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc*1000)/10
    
    return acc  



def confusionmatrix(y_pred1, y_test1, column=['normal','sis1','sis2','sis3','sis4']):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    # _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    a=[]
    b=[]
    for i in range(len(y_test1)):
        a.append(y_test1[i].detach().cpu().item())
        b.append(y_pred1[i].detach().cpu().item())
    y_test=a
    y_pred=b
    df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred))
    df_cm.index=column
    df_cm.columns=column
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    b, t = plt.ylim() 
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)
    

    df_cm =df_cm / df_cm.astype(np.float).sum(axis=1)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    b, t = plt.ylim() 
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)


def tsneplotafter(loader1):
    TSNEmodel = TSNE(learning_rate=100)

    xsource=[]
    ysource=[]
    # xtarget=[]
    # ytarget=[]

        
        
    for batch_idx, batch_data2 in enumerate(loader1):
            X_t,X_t2, X_s,X_s2, tlabel = batch_data2
            X_s = Variable(X_s.to(device))
            X_s2 = Variable(X_s2.to(device))
            s_r,s_f,s_feature = best_senc(X_s,X_s2)
            pred, s_log, s_features  = best_scls(s_feature)
        

            xsource.append(np.array(pred.detach().cpu()))
            ysource.append(tlabel.item())
    # for batch_idx, batch_data2 in enumerate(loader2):
    #         tinput, tlabel = batch_data2
    #         tinput = Variable(tinput.to(device))
    #         tinput = feature_extractor(tinput)
    #         xtarget.append(np.array(tinput.detach().cpu()))
    #         ytarget.append(tlabel.item())

    xsource=np.array(xsource)
    ysource=np.array(ysource)
    # xtarget=np.array(xtarget)
    # ytarget=np.array(ytarget)

#     for i in range(len(ytarget)):
#         if ytarget[i]==0:
#             ytarget[i]=5
#         elif ytarget[i]==1:
#             ytarget[i]=6
#         elif ytarget[i]==2:
#             ytarget[i]=7
#         elif ytarget[i]==3:
#             ytarget[i]=8
#         else:
#             ytarget[i]=9 
            
    xsource=xsource.reshape(ysource.size,-1)
    # xtarget=xtarget.reshape(ytarget.size,-1)
    ysource=ysource.reshape(-1,1)
    # ytarget=ytarget.reshape(-1,1)
    
    source=np.hstack([xsource,ysource])
    # target=np.hstack([xtarget,ytarget])
    # tsne=np.vstack([source,target])
    tsne2=source[source[:,-1].argsort()]

    source_normal=np.where(tsne2[:,-1] == 0)[0][0]
    source_sis1=np.where(tsne2[:,-1] == 1)[0][0]
    source_sis2=np.where(tsne2[:,-1] == 2)[0][0]
    source_sis3=np.where(tsne2[:,-1] == 3)[0][0]
    source_sis4=np.where(tsne2[:,-1] == 4)[0][0]
    
    # target_normal=np.where(tsne2[:,-1] == 5)[0][0]
    # target_sis1=np.where(tsne2[:,-1] == 6)[0][0]
    # target_sis2=np.where(tsne2[:,-1] == 7)[0][0]
    # target_sis3=np.where(tsne2[:,-1] == 8)[0][0]
    # target_sis4=np.where(tsne2[:,-1] == 9)[0][0]
    tsne_X = TSNEmodel.fit_transform(tsne2[:,:-1],tsne2[:,-1])

    plt.figure(figsize=(10,10))
    
    plt.scatter(tsne_X[:source_sis1,0],tsne_X[:source_sis1,1],c='red',marker='o', label = 'source normal')
    plt.scatter(tsne_X[source_sis1:source_sis2,0],tsne_X[source_sis1:source_sis2,1],c='blue',marker='o', label = 'source sis1')
    plt.scatter(tsne_X[source_sis2:source_sis3,0],tsne_X[source_sis2:source_sis3,1],c='yellow',marker='o', label = 'source sis2')
    plt.scatter(tsne_X[source_sis3:source_sis4,0],tsne_X[source_sis3:source_sis4,1],c='black',marker='o', label = 'source sis3')
    plt.scatter(tsne_X[source_sis4:,0],tsne_X[source_sis4:,1],c='purple',marker='o' ,label = 'source sis4')
    
#     plt.scatter(tsne_X[target_normal:target_sis1,0],tsne_X[target_normal:target_sis1,1],c='red',marker='s', label = 'target normal')
#     plt.scatter(tsne_X[target_sis1:target_sis2,0],tsne_X[target_sis1:target_sis2,1],c='blue',marker='s', label = 'target sis1')
#     plt.scatter(tsne_X[target_sis2:target_sis3,0],tsne_X[target_sis2:target_sis3,1],c='yellow',marker='s', label = 'target sis2')
#     plt.scatter(tsne_X[target_sis3:target_sis4,0],tsne_X[target_sis3:target_sis4,1],c='black',marker='s', label = 'target sis3')
#     plt.scatter(tsne_X[target_sis4:,0],tsne_X[target_sis4:,1],c='purple',marker='s', label = 'target sis4')

    plt.suptitle('tsne after')
    # plt.legend()

    plt.show()