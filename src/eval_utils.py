import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from src.models import *
import torch


CLASS_LABELS = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping and Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning and Coagulation",
    "Gallbladder Retraction",
]



def get_initial_prob(df):
    initial_prob_list = []
    initial_phase_df = df.groupby('video_num').nth(1).groupby('phase').count()
    for phase in df['phase'].unique():
        if phase in initial_phase_df.index:
            single_phase_prob = initial_phase_df[initial_phase_df.index == phase].iloc[:,0].values[0] / df['video_num'].unique().shape[0]
        else:
            single_phase_prob = 0
        initial_prob_list.append(single_phase_prob)

    return initial_prob_list

def model_select(model_num=0):
    '''
    Specify model num to select model to evaluate

    CNN's
    Model Num 0: ResNet18
    Model Num 1: ResNet18 LSTM
    Model Num 2: ResNet50 pretrained
    Model Num 3: ResNet50 pretrained + LSTM
    Model Num 4: Xception pretrained

    Transformer's
    Model Num 5: ViT
    Model Num 6: Cross ViT
    Model Num 7: BeiT
    Model Num 8: DeiT
    Model Num 9: DeiT OD
    Model Num 10: VIT OD
    '''
    model=None
    emission_df = None
    if model_num == 0:
        model_path = 'models/resnet_18_ord_model.pth'
        model = ResNet18Model(ResNetBlock)
        model.load_state_dict(torch.load(model_path))
        emission_df = pd.read_parquet('hmm_folder/resnet_18_ord_hmm_df.parquet')

    elif model_num == 1:
        model_path = 'models/resnet_18_lstm_ord_model.pth'
        model = ResNet18LSTM()
        model.load_state_dict(torch.load(model_path))
        emission_df = pd.read_parquet('hmm_folder/resnet_18_lstm_ord_hmm_df.parquet')

    elif model_num == 2:
        model_path = 'models/resnet_50_ord_model.pth'
        model = ResNet50()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/resnet_50_ord_hmm_df.parquet')

    elif model_num == 3:
        model_path = 'models/resnet_50_lstm_ord_model.pth'
        model = ResNet50LSTM()
        model.load_state_dict(torch.load(model_path))
        # model.freeze()
        emission_df = pd.read_parquet('hmm_folder/resnet_50_lstm_ord_hmm_df.parquet')
    elif model_num == 4:
        model_path = 'models/xception_ord_model.pth'
        model = Xception()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/xception_ord_hmm_df.parquet')

    elif model_num == 5:
        model_path = 'models/ViT_ord_model.pth'
        model = ViT()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/ViT_ord_hmm_df.parquet')

    elif model_num == 8:
        model_path = 'models/DeiT_ord_final_model.pth'
        model = DeiT_Distilled()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/DeiT_ord_final_hmm_df.parquet')
    # elif model_num == 8:
    #     model_path = 'models/Ordered_DeiT_model.pth'
    #     model = DeiT_Distilled()
    #     model.load_state_dict(torch.load(model_path))
    #     model.freeze()
        emission_df = pd.read_parquet('hmm_folder/Ordered_DeiT_hmm_df.parquet')

    elif model_num == 9:
        model_path = 'models/DeiT_ord_od_model.pth'
        model = DeiT_Distilled_OD()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/DeiT_ord_od_hmm_df.parquet')
    elif model_num == 6:
        model_path = 'models/Cross_ViT_ord_model.pth'
        model = Cross_Vit()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/Cross_ViT_ord_hmm_df.parquet')
    elif model_num == 7:
        model_path = 'models/BEIT_ord_model.pth'
        model = BEIT()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/BEIT_ord_hmm_df.parquet')
    elif model_num == 10:
        model_path = 'models/ViT_LSTM_ord_model.pth'
        model = ViT_LSTM()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/ViT_LSTM_ord_hmm_df.parquet')
    elif model_num == 11:
        model_path = 'models/DeiT_LSTM_ord_model.pth'
        model = DeiT_Distilled_LSTM()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/DeiT_LSTM_ord_hmm_df.parquet')
    # elif model_num == 11:
    #     model_path = 'models/DeiT_LSTM_ord_final_model.pth'
    #     model = DeiT_Distilled_LSTM()
    #     model.load_state_dict(torch.load(model_path))
    #     model.freeze()
    #     emission_df = pd.read_parquet('hmm_folder/DeiT_LSTM_ord_final_hmm_df.parquet')
    elif model_num == 12:
        model_path = 'models/resnet_50_od_phase_ord_model.pth'
        model = ResNet50_ST_Phase()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/resnet_50_od_phase_ord_hmm_df.parquet')
    elif model_num == 13:
        model_path = 'models/DeiT_od_LSTM_ord_model.pth'
        model = DeiT_Distilled_OD_LSTM()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/DeiT_od_LSTM_ord_hmm_df.parquet')
    elif model_num == 14:
        model_path = 'models/DeiT_LSTM_ord_32_model.pth'
        model = DeiT_Distilled_LSTM()
        model.load_state_dict(torch.load(model_path))
        model.freeze()
        emission_df = pd.read_parquet('hmm_folder/DeiT_LSTM_ord_32_hmm_df.parquet')
        
    return model, emission_df

def get_transition_matrix(vid_phase_array, n):

    phase_array = np.array(vid_phase_array)
    total_inds = phase_array.size - 1
    t_strided = np.lib.stride_tricks.as_strided(phase_array, shape=(total_inds,2),strides=(phase_array.strides[0],phase_array.strides[0]))  # type: ignore
    inds,counts = np.unique(t_strided, axis=0, return_counts=True)

    prob_mtrx = np.zeros((n,n))
    prob_mtrx[inds[:,0],inds[:,1]] = counts
    sums = prob_mtrx.sum(axis=1)

    prob_mtrx[sums!=0] = prob_mtrx[sums!=0] / sums[sums!=0][:,None]

    return prob_mtrx

# def show_ribbon_plot(values, title,classes=CLASS_LABELS):
#     fig = plt.figure(figsize=(12,3))
#     ax = fig.add_subplot(111)
#     ax.set_yticks([],[])
#     im = ax.pcolormesh([values], vmin=0, vmax=7)
#     cbar = fig.colorbar(im,orientation='vertical',drawedges=True)
#     cbar.ax.set_yticks(np.arange(len(classes)))
#     cbar.ax.set_yticklabels(classes)
#     ax.set_title(title)
#     ax.set_xlabel('Video Timestamp')
#     fig.tight_layout()
#     plt.show()

def show_ribbon_plot(values, title,classes=CLASS_LABELS):
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    ax.set_yticks([],[])
    colour_list = [
        (0.267004, 0.004874, 0.329415, 1.0),
        (0.275191, 0.194905, 0.496005, 1.0),
        (0.212395, 0.359683, 0.55171, 1.0),
        (0.153364, 0.497, 0.557724, 1.0),
        (0.122312, 0.633153, 0.530398, 1.0),
        (0.288921, 0.758394, 0.428426, 1.0),
        (0.626579, 0.854645, 0.223353, 1.0)]
    cmap = colors.ListedColormap(colour_list)
    im = ax.pcolormesh([values],cmap=cmap, vmin=0, vmax=7)
    custom_lines = [
        Line2D([], [], marker='|', color=colour_list[i], linestyle='None', markersize=10, markeredgewidth=5) for i in range(len(colour_list))]
    ax.legend(custom_lines, classes, bbox_to_anchor=(1.30, 0.9))
    ax.set_title(title)
    ax.set_xlabel('Video Timestamp')
    fig.tight_layout()

    plt.show()     


def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])


def fix_length_preds(pred_array,cnn_probs,thresh=30):
    new_pred_array = pred_array.copy()
    if len(pred_array) >= 2500:
        thresh = 40
    if len(pred_array) <= 100:
        thresh = 20

    

    lengths,positions,phase_labels = rle(new_pred_array)
    smaller_than_thresh = lengths[1:-1] < thresh
    
    while any(smaller_than_thresh):
        length_to_fix = lengths[1:-1][smaller_than_thresh]
        positions_to_fix = positions[1:-1][smaller_than_thresh]
        phase_labels = phase_labels[1:-1][smaller_than_thresh]
 
        #Find preivous section predictions
        final_positions_start = positions_to_fix - 1
        final_positions_end = positions_to_fix + length_to_fix + 1
        #Check if they are the same
        eq_prev_preds = new_pred_array[final_positions_start] == new_pred_array[final_positions_end]
        for i in range(len(positions_to_fix[eq_prev_preds])):
            new_pred_array[positions_to_fix[eq_prev_preds][i]:positions_to_fix[eq_prev_preds][i] + length_to_fix[eq_prev_preds][i]] = new_pred_array[final_positions_start][eq_prev_preds][i]

        if not any(eq_prev_preds):
            for j in range(len(positions_to_fix[~eq_prev_preds])):
                cnn_comp_probas = (np.vstack(cnn_probs[positions_to_fix[j]:positions_to_fix[j] + length_to_fix[j]]) / np.linalg.norm(np.vstack(cnn_probs[positions_to_fix[j]:positions_to_fix[j] + length_to_fix[j]]))).clip(min=0)
                cnn_sum_sorted = np.argsort(cnn_comp_probas.sum(axis=0))[::-1]
                likeliest_class = new_pred_array[positions_to_fix[j]]
                for phase_class in cnn_sum_sorted:
                    if phase_class == new_pred_array[final_positions_start][~eq_prev_preds][j] or phase_class == new_pred_array[final_positions_end][~eq_prev_preds][j]:
                        likeliest_class = phase_class
                        break
                new_pred_array[positions_to_fix[~eq_prev_preds][j]:positions_to_fix[~eq_prev_preds][j] + length_to_fix[~eq_prev_preds][j]] = likeliest_class
                    
   
        lengths,positions,phase_labels = rle(new_pred_array)
        smaller_than_thresh = lengths[1:-1] < thresh
     
    return new_pred_array



def single_vid_eval(results_df, hmm_model,ma_length=50,show_plots=True,classes=CLASS_LABELS,method='raw'):
    vid_name = results_df['video_num'].iloc[0]
    print(f"Start decoding vid {vid_name}:  ...")
    ground_truth = results_df['true_labels'].to_numpy()
    vision_prediction = results_df['predicted_labels'].to_numpy()
    ma_probs = np.array([np.array(xi) for xi in results_df['cnn_output'].to_numpy()])
    vision_probs = np.array([np.array(xi) for xi in results_df['cnn_output'].to_numpy()])


    #Apply moving average filtering on specified number of frames
    for i in range(1, ma_length):
        ma_probs[i:] = (i * ma_probs[i:] + ma_probs[:-i]) / (i + 1)

    vision_prediction_smooth = np.argmax(np.array(ma_probs), axis=1)  
    
    # Get HMM prediction
    # Decode the sequence
    (_, hmm_predict) = hmm_model.decode(vision_probs, algorithm="viterbi")

    hmm_predict_fixed = fix_length_preds(hmm_predict,vision_probs)

    # print("Vision only")
    # print(classification_report(ground_truth, vision_prediction))

    # print("Vision + Moving Average")
    # print(classification_report(ground_truth, vision_prediction_smooth))
    
    # print("Vision+HMM")
    # print(classification_report(ground_truth, hmm_predict))

    
    
    if show_plots == True:
        show_ribbon_plot(ground_truth,'Ground Truth',classes)

        show_ribbon_plot(vision_prediction,'Vision Only',classes)

        # show_ribbon_plot(vision_prediction_smooth,'Vision + Moving Average',classes)

        show_ribbon_plot(hmm_predict, 'Vision+HMM',classes)

        show_ribbon_plot(hmm_predict_fixed, 'Vision+HMM+Postprocesing',classes)




    #Save HMM metrics for video
    if method == 'raw':
        prec =  precision_score(ground_truth, vision_prediction,average='weighted')
        rec = recall_score(ground_truth, vision_prediction,average='weighted')
        f1 = f1_score(ground_truth, vision_prediction,average='weighted')
        accuracy = accuracy_score(ground_truth, vision_prediction) 
        class_rep = classification_report(ground_truth, vision_prediction) 
        predict = vision_prediction
    
    elif method == 'ma':
        prec =  precision_score(ground_truth, vision_prediction_smooth,average='weighted')
        rec = recall_score(ground_truth, vision_prediction_smooth,average='weighted')
        f1 = f1_score(ground_truth, vision_prediction_smooth,average='weighted')
        accuracy = accuracy_score(ground_truth, vision_prediction_smooth) 
        class_rep = classification_report(ground_truth, vision_prediction_smooth)
        predict = vision_prediction_smooth
    elif method == 'hmm':
        prec = precision_score(ground_truth, hmm_predict,average='weighted')
        rec = recall_score(ground_truth, hmm_predict,average='weighted')
        f1 = f1_score(ground_truth, hmm_predict,average='weighted')
        accuracy = accuracy_score(ground_truth, hmm_predict)
        class_rep = classification_report(ground_truth, hmm_predict)
        predict = hmm_predict

    else:
        prec = precision_score(ground_truth, hmm_predict_fixed,average='weighted')
        rec = recall_score(ground_truth, hmm_predict_fixed,average='weighted')
        f1 = f1_score(ground_truth, hmm_predict_fixed,average='weighted')
        accuracy = balanced_accuracy_score(ground_truth, hmm_predict_fixed)
        class_rep = classification_report(ground_truth, hmm_predict_fixed)
        predict = hmm_predict_fixed


    return prec, rec,f1,accuracy, class_rep, ground_truth, predict