## About
Summary Transformer-based architectures for Computer Vision. Focusing on object detection, segmentation, and few-shot segmentation.

Keep updated

## Table of contents
* [Standard Transformer](#Standard-Transformer)
	* [Original Transformer](#Original-Transformer)
	* [ViT (Vision Transformer)](#ViT-Vision-Transformer)
	* [VTN (Video Transformer Network)](#VTN-Video-Transformer-Network)
	* [ViTGAN](#VitGAN)
* [Object Detection/Segmentation Transformer](#Object-DetectionSegmentation-Transformer)
	* [DETR (Detection Transformer)](#DETR-Detection-Transformer)
	* [AnchorDETR](#AnchorDETR)
	* [MaskFormer](#MaskFormer)
	* [TransUNet](#TransUNet)
	* [SegFormer](#SegFormer)
	* [Fully Transformer Networks (FTN)](#Fully-Transformer-Networks-FTN)
	* [SOTR (Segmenting Objects with Transformer)](#SOTR-Segmenting-Objects-with-Transformer)
	* [UTNet](#UTNet)
	* [HandsFormer](#HandsFormer)
* [Few-shot Transformer](#Few-shot-transformer)
	* [Meta-DETR: Image-Level Few-Shot Object Detection with Inter-Class Correlation Exploitation](#Meta-DETR-Image-Level-Few-Shot-Object-Detection-With-Inter-Class-Correlation-Exploitation)
	* [Boosting Few-shot Semantic Segmentation with Transformers](#Boosting-Few-shot-Semantic-Segmentation-with-Transformers)
	* [Few-Shot Segmentation via Cycle-Consistent Transformer](#Few-Shot-Segmentation-via-Cycle-Consistent-Transformer)
	* [Few-shot Semantic Segmentation with Classifier Weight Transformer](#Few-shot-Semantic-Segmentation-with-Classifier-Weight-Transformer)
	* [Few-shot Transformation of Common Actions into Time and Space](#Few-shot-Transformation-of-Common-Actions-into-Time-And-Space)
* [Other](#Other)
	* [Do Vision Transformer see like CNN](#Do-Vision-Transformer-see-like-CNN)
	* [Unifying Global-Local Representations in Salient Object Detection with Transformer](#Unifying-Global-Local-Representations-in-Salient-Object-Detection-with-Transformer)
* [Resource](#Resource)

## Standard Transformer

### Original Transformer
+ **Paper**: https://arxiv.org/abs/1706.03762
![](Images/Transformer.png)  
+ **Input**: 
	- Sequence embedding (e.g. word embeddings of a sentence)
	- Positional Encoding => encode the _positions of embedding word within the sentence_ in the input of Encoder/Decoder block
+ **Encoder**:
	- Embedding words => Skip_Connection[**_MHSA_** => Norm] => Skip_Connection[**_FFN_** => Norm] => Encoder output
		- **_MHSA_**: Multi-Head Self Attention
		- **_FFN_**: FeedForward Neural Network
	+ Repeat N times (N usually 6)
+ **Decoder**:
	- Decoder input:
		- Leaned output of the decoder (initial token in the begining, learned sentence throughout the process)
		- Encoder input (put in the middle of the Decoder)
	- (Input + Positional Encoding) => Skip_Connection[**_MHSA_** + Norm] => Skip_Connection[(+Encoder input) => **_MHSA_** => Norm] => Skip_Connection[**_FFN_** + Norm] => Linear => Softmax => Decoder Output
	- Using the decoder output as the input for next round, repeat N times (N ussually 6)
+ [Read here](https://github.com/quanghuy0497/Deep-Learning-Specialization/tree/main/Course%205%20-%20Sequence%20Models#transformer-network-1) for more detail
	+ Also [here](https://phamdinhkhanh.github.io/2019/06/18/AttentionLayer.html) if you prefer  detailed explanation in Vienamese
+ **Codes**: https://github.com/SamLynnEvans/Transformer

### ViT (Vision Transformer)
+ **Paper**: https://arxiv.org/pdf/2010.11929.pdf
![](Images/ViT.gif)  
+ **Input**:
	-  Image [H, W, C] => non-overlapped patches (conventionally 16x16 patch size) => flatten into sequence => linear projection (vectorized + Linear) => **_patch embeddings_**
	- **_Positional encoding_** added to the patch embeddings for location information of the patchs sequence
	- Extra learnable `[Cls]` token (embedding) + positional 0 => attached on the head of the embedding sequence (denote as `Z0`)
+ **Architecture**: (Patch + Position Embedding) => Transformer Encoder => MLP Head for Classification
	-  **_Transformer Encoder_**: Skip_Connection[Norm => **_MHSA_**] => Skip_Connection[Norm + **_MLP_**(Linear, GELU, Linear)] => output
	- **_MLP Head for classification_**:  `C0` (output of `Z0` after went through the Transformer Encoder) => **_MLP Head_** (Linear + Softmax) => classified label
+ **Good video explanation**: https://www.youtube.com/watch?v=HZ4j_U3FC94
+ **Code**: https://github.com/lucidrains/vit-pytorch

### VTN (Video Transformet Network)
+ **Paper**: https://arxiv.org/pdf/2102.00719.pdf  
![](Images/VTN.png)  
+ Based on [Longformer](https://arxiv.org/pdf/2004.05150.pdf) (transformer-based model can process a long sequence of thousands of tokens)
+ **Pipeline**: Quite similar to the standard ViT
	- **_The 2D spatial backbone_** `f(x)` can be replaced with any given backbone for 2D images => feature extraction
	- **_The temporal attention-based encoder_** can stack up more layers, more heads, or can be set to a different Transformers model that can process long sequences. 
		- Note that a special classification token `[CLS]` is added in front of the feature sequence => final representation of the video => classification task head for video classifies
	- **_The classification head_** can be modified to facilitate different video-based tasks, i.e. temporal action localization
+ **Code**: https://github.com/bomri/SlowFast/tree/master/projects/vtn

### ViTGAN
+ **Paper**: https://arxiv.org/pdf/2107.04589.pdf  
![](Images/ViTGAN.png)  
+ Both the generator and the discrimiator are designed based on the stadard ViT, but with modifications
+ **Architecture**:
	+ **_Generator_**: Input latent `z` => _Mapping Network_ (MLP) => latent vector `w` => _Affine transform_ `A` => _Transformer Encoder_ => _Fourier Encoding_ (sinusoidal/sine activation) => `E_fou` => _2-layer MPL_ => Generated Patches
		- **_Transformer Encoder_**: 
			- Embedding position => Skip_Connection[SLN => MHSA] => Skip_Connection[SLN =>MLP] => output
			- SLN is called **_Self-modulated LayerNorm_** (as the modulation depends on no external information)
				- The SLN formula is describe within the paper (Equation 14)
			- Embedding `E` as initial input; `A` as middle inputs for the norm layers
	+ **_Discriminator_**: the pipeline architecture is similar to the standard ViT model, but with several changes:
		- Adapt the overlapping patches at the begining (rather than the nonoverlapping ones)
		- Replace the dot product between `Q` and `K` with Euclidean (L2) distance in the Attention formula
		- Apply spectral normalization (read paper for more information)
+ **Code**: To be updated

## Object Detection/Segmentation Transformer

### DETR (Detection Transformer)
+ **Paper**: https://arxiv.org/pdf/2005.12872.pdf  
![](Images/DETR.png)  
+ **Transformer Encoder & Decoder**:
	- Share the same architecture with the original transformer
	- *Encoder:*
		- Input sequence: flattened 2D feature (Image => CNN => flatten) + learnable fixed positional encoding (add to each layer)
		- Output: encoder output in sequence
	- *Decoder:*
		- Input: Object queries (learned positional embeddings) + encoder output (input in the middle)
		- Output: output embeddings
		- The Decoder decode N objects in parallel at each decoder layer, not sequence one element at a time
		- The model can reason about all objects together using pair-wise relations between them, while being able to use whole image as content
+ **Prediction FFN**:
	- 3-layer MLP with ReLU  
	- Output embeddings as input
	- Predict normalized center coordinates, heigh and width of bounding box  
+ **Architecture**:
	Image => **_Backbone (CNN)_** => 2D representation => Flatten (+ Positional encoding) => **_Transformer Encoder-Decoder_** => **_Prediction FFN_** => bounding box
+ **Code**: https://github.com/facebookresearch/detr

### AnchorDETR
+ **Paper**: https://arxiv.org/pdf/2109.07107.pdf  
![](Images/AnchorDETR.png)  
+ **Backbone**: ResNet40
+ The encoder/decoder share the same structure as DETR
	- However, the self-attention in the encoder and  decoder blocks are replaced by Row-Column Decouple Attention
+ **Row-Column Decouple Attention**:
	- Help reduce the GPU memeory when facing with high-resolution feature
	- Main idea:
		- Decouple key feature `Kf` into row feature `Kf,x` and column feature `Kf,y` by 1D global average pooling
		- Then perform the row attetion and column attentions speparately
+ **Code**: https://github.com/megvii-research/AnchorDETR

### MaskFormer
+ **Paper**: https://arxiv.org/pdf/2107.06278.pdf  
![](Images/MaskFormer.png)  
+ **Pixel-level module**:
	- Image => **_Backbone (ResNet)_** => image feature `F` => **_Pixel decoder (upsampling)_** => per-pixel embedding `E_pixel`
+ **Transformer module (Decoder only)**:
	- Standard Transformer decoder
	- N queries (learneable positional embeddings) + `F` (input in the middle) => **_Tranformer Decoder_** => N per-segment embeddings `Q`
	- Prediction in parallel (similar to DETR)
+ **Segmentation module**:
	- `Q` => *MLP (2 Linears + solfmax)* => N mask embeddings `E_mask` & N class predictions
	- *Dot_product*(`E_mask`, `E_pixel`) => sigmoid => Binary mask predictions
	- *Matrix_mul*(Mask predictions, class predictions) => Segmantic segmentation
+ **Code**: https://github.com/facebookresearch/MaskFormer

### TransUNet
+ **Paper**: https://arxiv.org/pdf/2102.04306.pdf  
![](Images/TransUNet.png)  
+ **Downsampling (Encoder)**: using CNN-Transformer Hybrid
	+ (Medical) Image `[H, W, C]` => _**CNN**_ => 2D feature map => _**Linear Projection**_ (Flatten into 2D Patch embedding) => Downsampling => _**Tranformer**_ => Hidden feature `[n_patch, D]`
		- CNN: downsampling by 1/2 => 1/4 => 1/8
		- Transformer: Norm layer *before* MHSA/FFN (rather than applying Norm layer after MHSA/FFN like the original Transformer), total 12 layers
	+ Why using CNN-Transformer hybrid:
		- Leverages the intermediate high-resolution CNN feature maps in the Decoder
		- Performs better than the purge transformer
+ **Upsamling (Decoder)**: using Cascaded Upsampler 
	- Similar to the upsamling part of the [standard UNet](https://github.com/quanghuy0497/Deep-Learning-Specialization/tree/main/Course%204%20-%20Convolutional%20Neural%20Networks#u-net-architecture)
		- **_Upsampling_** => *concat* with corresponded CNN feature map (from the Encoder) => *Conv3x3 with ReLu*
		- **_Segmentation head_** (Conv1x1) at the final layer
	- Hidden Feature `[n_patch, D]` => reshape `[D, H/16, W/16]` => `[512, H/16, H/16]` => `[256, H/8, W/8]` => `[128, H/4, W/4]` => `[64, H/2, W/2]` => `[16, H, W]` => Segmentation head => Segmantic Segmentation
+ **Code**: https://github.com/KenzaB27/TransUnet

### SegFormer
+ **Paper**: https://arxiv.org/pdf/2105.15203.pdf  
![](Images/SegFormer.png)  
+ **Input**: Image `[H, W, 3]` => patches of size 4x4 (rather than 16x16 like ViT)
	- Using smaller patches => favor the dense prediction task
	- Do not need positional encoding (PE):
		- Not necessary for semantic segmentation
		- The resolution of PS is fixed => needs to be interpolated when facing different test resolutions => dropped accuracy
+ **Hierarchical Transformer Encoder**: extract coarse and fine-grained features, partly inspired by ViT but optimized for semantic segmentation
	- Overlap patch embeddings => [**_Transformer Block_** => Downsampling] x 4 times => CNN-like multi-level feature map `Fi`
		- Feature map size: `[H, W, 3]` => `F1` `[H/4, W/4, C1]` => `F2` `[H/8, W/8, C2]` => ... => `F4` `[H/32, W/32, C4]`
			- Provide both high and low-resolution features => boost the performance of semantic segmentation
		- Transformer Block1: Efficient Self-Atnn => Mix-FNN => Overlap Patch Merging
			- **_Efficient Self-Attention_**: Use the reduction ratio R on sequence length N = H x W (in particular, apply stride R) => reduce the complexity by R times
			- **_Mix-FFN_**: Skip_Connection[MLP => Conv3x3 => GELU => MLP] which considers the effect of zero padding to leak location information (rather than positional encoding)
			- **_Overlapped Patch Merging_**: similar to the image patch in ViT but overlap => combine feature patches
+ **Lightweight All-MLP Decoder**: fuse the multi-level features => predict semantic segmentation mask
	1. **_1st Linear layer_**: unifying channel dimension of multi-level features `Fi` (from the encoder)  
	2. `Fi` are **_upsampler_** to 1/4th and **_concat_** together  
	3. **_2nd Linear layer_**: fusing concatenated features `F`
	4. **_3rd Linear layer_**: predicting segmentation mask M `[H/4, W/4, N_cls]` with `F`
+ **Code**: https://github.com/lucidrains/segformer-pytorch

### Fully Transformer Networks (FTN)
+ **Paper**: https://arxiv.org/pdf/2106.04108.pdf  
![](Images/FTN.png)  
+ Fully Transformer Networks for semantic image segmentation, without relying on CNN.
	- Both the encoder and decoder are composed of multiple transformer modules
	- **_Pyramid Group Transformers (PGT) encoder_** to divide feature maps into multiple spatial groups => compute the representation for each
		- Capable to handle spatial detail or local structure like CNN
		- Reduce unaffordable computational & memory cost of the standard ViT; reduce feature resolution and increase the receptive field for extracting hierarchical features
	- **_Feature Pyramid Transformer (FPT) decoder_** => fuse semantic-level & spatial level information from PGT encoders => high-resolution, high-level semantic output
+ **Architecture**:
	- Image => Patch => PGT Encoder => FPT Decoder => linear layer => bilinear upsampling => probability map => argmax(prob_map) => Segmentation
	- **_PGT_**: four hierarchical stages that generate features with multiple scales, include Patch Transform (non-overlapping) + PGT Block to to extract hierarchical representations
		+ PGT Block: Skip_Connection[Norm => PG-MSA] => Skip_Connection[Norm => MLP]
		+ **_PG-MSA (Pyramid-group transformer block)_**: `Head_ij`=Attention(Qij,Kij,Vij) => `hi` = reshape(Head_ij) => PG-MSA = Concat(`hi`)
	- **_FPT_**: aggregate the information from multiple levels of PGT encoder => generate finer semantic image segmentation
		+ The scale of FPT is not larger the better for segmentation (with limited segmentation training data) => determined by depth, embedding dim, and the reduction ratio of SR-MSA 
		+ **_SR-MSA (Spatial-reduction transformer block)_**: reduce memory and computation cost by spatially reducing the number of Key & Value tokiens, especially for high-resoluton representations
		+ The multi-level high-resolution feature of each branch => fusing (element-wise summation/channel-wise concatenation) => finer prediction
+ **Code**: To be updated

### SOTR (Segmenting Objects with Transformer)
+ **Paper**: https://arxiv.org/pdf/2108.06747.pdf  
![](Images/SOTR.png)  
+ Combines the advantages of CNN and Transformer
+ **Architecture**:
	+ **_Pipeline_**:
		- Image => _CNN Backbone_ => feature maps in multi-scale => patch recombination + positional embedding => clip-lvel feature sequences/blocks => _Transformer_ => global-level semantic feature => functional heads => class & conv kernel prediction 
		- Backbone output  => _Multi-level upsampling model_ (with dynamic conv) => dynamic conv(output, Kerner head) => instance masks
	+ **_CNN Backbone_**: Feature pyramid network
	+ **_Transformer_**: proposed 2 different transformer designs with Twin attention:
		- _Twin attention_: simplify the attention matrix with sparse representation (as the self-attention has both quadratic time and memory complicity) => higher computational cost
			- Calculate attention within each column (independent between columns) => calculate attention within each row (independent between rows) => connect together
			- Twin att. has a global receptive field & covers the information along 2 dimension
		![](Images/Twin_Att.png)  
		- Transformer layer:
			- _Pure twin layer_: Skip_Connection[Norm => Twin Att.] => Skip_Connection[Norm => MLP]
			- _Hybrid twin layer_: Skip_Connection[Norm => Twin Att.] => Skip_Connection[Conv3x3 => Leaky ReLU => Conv3x3]
			- Hybrid Twin comes with the best performance
	+  **_Multi-level upsampling model_**: P5 feature map + Positional from transformer + P2-P4 from FPN => [Conv3x3 => Group Norm => Relu, multi stage] => upsample x2, x4, x8 (for P3-P5) => added together => point-wise conv => upsamping => final `HxW` feature map
+ **Code**: https://github.com/easton-cau/SOTR

### UTNet
+ **Paper**: https://arxiv.org/pdf/2107.00781.pdf  
![](Images/UTNet.png)
+ **Pipeline**:
	- Apply conv layers to extract local intensity feature, while using self-attention to capture long-range associative information
	- UTNet follows the standard design of UNet, but replace the last conv of the building block in each resolution (except the highest one) with the proposed Transformer module
	- Rather than using the convention MHSA like the standard Transformer, UTNet develops the _Efficient MHSA_ (quite similar to the one in SegFormer):
		- **_Efficient MHSA_**: Sub-sample `K` and `V` into low-dimensional embedding (reduce size by 8) using Conv1x1 => bilinear interpolation 
		![](Images/Efficient_MHSA.png)
	- Using 2-dimensional relative position encoding by adding relative height and width information rather than the standard position encoding
+ **Code**: https://github.com/yhygao/UTNet

### HandsFormer
+ **Paper**: https://arxiv.org/pdf/2104.14639.pdf  
![](Images/HansFormer.png)  
+ **Architecute**:
	- Image => _UNet_ => Image features (from layers of UNet decoder) + Keypoint heatmap => _bilinear interpolation & concat_ => _FFN (3-layer MLP)_ => _concat_ with keypoint heatmap (with positional encoding) => keypoint representation (likely to correspond to the 2D location of hand joints)
		- Localizing the joints of hands in 2D is more accurate than directly regressing 3D location.
		- The 2D keypoints are a very good starting point to predict an accurate 3D pose for both hands
	- Keypoint representation => _Transformer encoder_ => FFN (2-layer MLP + linear projection) => _Keypoint Identity predictor_ [2 FC layer => linear projection => Softmax] => 2D pose 
	- [Joint queries, Transform encoder output] => _Transform decoder_ => 2-layer MLP + linear projection => 3D pose
+ **Code**: To be updated

## Few-shot Transformer

### Meta-DETR: Image-Level Few-Shot Object Detection with Inter-Class Correlation Exploitation
+ **Paper**: https://arxiv.org/pdf/2103.11731.pdf  
![](Images/Meta-DETR.png)
+ Employ Deformable DETR and original Transformers as basic detection framework
+ **Architecture**:
	- Query image, Support images => Feature Extractor(ResNet-101) => Positional Encoding => Query/Support features => **_Correlational Aggregation Module_** => Support-Aggregated Query Feature => **_Transformer Encoder-Decoder_** => **_Prediction Head_** => Few-shot detection
+ **Correlation Aggregation Module (CAM)**:
	![](Images/CAM.png)  
	- Key-compoment in Meta-DETR => aggregates query features with support classes => class-agnostic prediction
		- Can aggregate multiple support classes simultaneously => capture inter-class correlations => reduce misclassification, enhance generalization
	- Pipeline:
		- Query & Support features => MHSA => ROIAlign + Average pooling (on the support feature only) => Query feature map `Q` & Support prototypes `S` 
		- `S` = Concat(`S`, BG-Prototype); Task Encodings `T` = Concat(`T`, BG-Encoding)
		- [`Q`, `S`, `T`] => Feature maching & Encoding matching (in parallel) => FFN => Support-Aggregated Query Features
	- **_Feature matching_**:
		- [`Q`, Sigmoid(`S`), `S`] => Single-Head Attention => Element-wise multiplication => feature matching output `Qf`
	- **_Encoding Matching_**:
		- [`T`, `Q`, `S`] => Single-Head Attention => encoding output `Qe`
	- **_FFN_**:
		Element-wise Add(`Qf`, `Qe`) => FFN => Support-Aggregated Query features
+ **Transformer Encoder-Decoder**:
	- Follow the [Deformable DETR](https://arxiv.org/pdf/2010.04159.pdf) architecture with 6 layer
		- Adapt Multi-scale deformable attention, with the CAM is counted as one encoder layer
	- Support-Aggregated Query features => Encoder-Decoder => Embedding `E`
+ **Prediction Head**:
	- `E` => FC, Sigmoid => Confidence score
	- `E` => FC, ReLU + FC, ReLU + FC, Sigmoid => Bounding Box
+ **Code**: https://github.com/ZhangGongjie/Meta-DETR

### Boosting Few-shot Semantic Segmentation with Transformers
+ **Paper**: https://arxiv.org/pdf/2108.02266.pdf  
![](Images/Boosting_FS_Segment.png)
+ **Architecture**:
	- Support/Query Images => Backbone (VGG/ResNet) => PANet => PFENet => `X`
		- **_PANet_**: Computer similarity between query features and support prototypes
		- **_PFENet_**: Concat(Query features; Expanded Support Prototypes; Prior Mask)
	- `X` => **_Multi-scale Processing_** => **_Global/Local Enhancement Module_** (with 2 ouput `T` and `Z` respectively) => Concat(T, Z) => MLP => Few-shot Semantic Segmentation
+ **Multi-scale Processing**:
	- Information over different scales (of the input feature maps `X` from support/query images) can be utilized
	- Multi-scaling with Global Average Pooling => feature Pyramid `Xi` = {X1, X2,...,Xn}
+ **Global Enhancement Module (GEM)**:
	- Using **_Transformer_** => enhance the feature to exploit the global information
	- `Xi` => FC layer (for channel reduction) => `X'i` => **_Feature Merging Unit_** (FMU) => `Yi`
		- FMU: 
			- `Yi` =  `X'i` if i=1; 
			- `Yi` =  (Conv1x1(*Concat*(`X'i`, `Ti-1`)) + `X'i`) if i>1
	- `Yi` => [**_MHSA_** => MLP (2 Linear)] => MHSA => `Ti` => `T`
		- MHSA with GELU and Norm
		- [MHSA => MLP] repeat L times with L = 3
		- `T` = = *Concat*(T1, T2,...Tn) at layer L
+ **Local Enhancement Module (LEM)**:
	- Follow the same pipeline as GEM:
		- `Xi` => FC layer => FMU => `Yi` => Conv => output
	- Rather than using transformer in GEM, LEM using **_Convolutional_** => encode the local information
	- LEM output: {Z1, Z2,...,Zn}  
+ **Segmentation Mask Prediction**:
	- Local/Global output: `Z` = *Concat*(Z1, T2,...Tn)
	- `Z` => MLP => target mask `M`
+ **Code**: https://github.com/GuoleiSun/TRFS

### Few-Shot Segmentation via Cycle-Consistent Transformer
+ **Paper**: https://arxiv.org/pdf/2106.02320.pdf  
![](Images/CyCTR.png)  
+ **Architecture**:
	+ Query & Support Images => Backbone (ResNet) => Image features `Xq` and `Xs` => Concat the mask averaged support feature to `Xq` and `Xs` => Flatten into 1D sequence by Conv1x1 (with shape HW x D) => (**_CyC-Transformer_**, L times) => reshaping => **_Conv-head_** => segmentation mask
	+ `Xq` and `Xs` has token represented by feature `z` at on pixel location => beneficial for segmentation
+ **CyC-Transformer**:
	+ **_Self-Alignment block_**:
		- Just like the original Transformer encoder
		- Flatten query feature as input (query only)
		- Pixel-wise feature of query images => aggregate their global context information
	+ **_Cross-alignment block_**:
		- Replace MHSA with **_CyC-MHA_**
		- Flatten query feature and sample of support feature as input
		- Performs attention between query and support pixel-wise features => aggregate relevant support feature into query ones
	+ **_Cycle-Consistent Multi-head Attention (CyC-MHA)_**:
		- Alleviate the excessive harmful support features that confuse pure pixel-level attention
		- Pipeline:
			-  Affinity map A is calculated => measure the correspondence relationship between all query and support tokens
			- For a single token position j (j={0,...Ns}), its most similar point `i*` = argmax A(`i`,`j`) with i={0,...HqWq} is the index of flatten query feature
			- Construct **_Cycle-consistency_** (CyC) relationship for all tokens in the support sequence
				- The cycle-consistency help avoids being bias by possible harmful feature effectively (facing when training for few-shot segmentation)
			- Finally, CyC-MHA = softmax(Ai + B)V
				- Where B (with only 2 value -inf and 0) is the additive bias element-wise added to aggregate support feature and V is the value sequence
		- With B, the attention weight tends to be zero => irrelevant information will not be consider
		- CyC encourages the consistency between most relative features between query and support => produce consistent feature representation
+ **Conv-head**:
	- Output of CyC-MHA => reshaping to spatial dimensions => Conv-Head (Cov3x3 => ReLu => Conv1x1) => Segmentation Mask
+ **Code**: To be updated

### Few-shot Semantic Segmentation with Classifier Weight Transformer
+ **Paper**: https://arxiv.org/pdf/2108.03032.pdf  
![](Images/CWT.png)  
+ **Architecture**:
	- **_First stage_**: pre-train encoder/decoder (**PSPNet** pre-trained on ImageNet) with supervised learning => stronger representation
		- Support/Query Image => Encoder-Decoder (**PSPNet**) => Linear clasifier with Support Mask (Support only) => Classifier (Support only) => [Classifier weight `Q`, Query feature `K`, Query feature `V`]
	- **_Second stage_**: meta-train the Classifier Weight Transformer (CWT) only (as the encoder-decoder capapble to capture generalization of unseen class)
		- [`Q`,`K`,`V`] => Skip_Connection[Linear => MHSA => Norm] => Conv operation with `Q` => Prediction Mask
+ **Code**: https://github.com/zhiheLu/CWT-for-FSS

### Few-shot Transformation of Common Actions into Time and Space
+ **Paper**: https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Few-Shot_Transformation_of_Common_Actions_Into_Time_and_Space_CVPR_2021_paper.pdf  
![](Images/Few_shot_Transformer.png)  
+ The goal is localize the spatio-temporal tubelet of an action in an _untrimmed query video_ based on the common action in the _trimment support video_
+ **Architecture**:  
	+ **_Pipeline_**:
		- Untrimmed query video => split into clips + few support video => _Video feature extractor_ (I3D) => spatio-temporal representation => _Common Attention block_ => aligned with previous clip feature => query clip feature 
		- Query clip feature => _Few-shot Transformer (FST)_ => fuse the support features into the query clip feature => aggregating with input embedding
		- Top of output embedding (from FST) => _Precition network_ => final tubelet prediction
	+ **_Video feature extractor_**:
		- Adapt I3D as backbone to obtain spatio-temporal representation of a single query video & few support videos
		- Support video => feed the whole video to the backbone directly
		- Untrimmed query video => split into multiple clips => backbone network
		- **_Common attention block_**: built on self-attention mechanism & non-local structural => model long-term spatio-temporal information
			- Formula: `A^(I1,I2) = I1 + Linear(Norm[A(I1,I2)])`
				- With A^ is the common attention, A is the standard atteention, I1, I2 are 2 inputs
			- Common attention aligns each query clip feature with its previous clip features => contain more motion information => benefit the common action localization
	+ **_Few-shot Transformer (FST)_**:    
	![](Images/Few_shot_Transformer_detailed.png)   
		- _Encoder_: standard architecture with MHSA. The input is supplied with fixed spatio-temporal positional encoding.
			- Support branch: the support video => encoder one by one => concat => `Es` => decoder (along with `Eq`)
			- Quary branch: enhanced query clip => encoder => `Eq` => decoder
			- The FFN can be a Conv1x1
		- _Decoder_: 3 input [`Es`, `Eq`, input embedding (learnt positional encoding)] => Common attention/MHSA => MHSA => FFN => Add&Norm => output embedding
	+ **_Prediction network_**: output embedding => 3-layer FFN with ReLU => linear projection => normalized center coordinates => final action tubes for the whole untrimmed query video
+ **Code**: To be updated

## Other:

### Do Vision Transformer see like CNN 
+ **Paper**: https://arxiv.org/pdf/2108.08810.pdf

### Unifying Global-Local Representations in Salient Object Detection with Transformer
+ **Paper**: https://arxiv.org/pdf/2108.02759.pdf

## Resource
+ Papers collection about Transformer in Computer Vision: 
	- https://github.com/dk-liang/Awesome-Visual-Transformer
	- https://github.com/DirtyHarryLYL/Transformer-in-Vision
	- https://github.com/Yangzhangcst/Transformer-in-Computer-Vision






<br><br>
<br><br>
These notes were created by [quanghuy0497](https://github.com/quanghuy0497/Transformer4Vision)@2021
