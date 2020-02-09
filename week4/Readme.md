#### Here is the step-by-step approach, being taken to achieve the required goals (meeting 99.4%+ accuracy, under the required parameters: <=20K and within 20 epochs) :
 
Fix up the required architecture/layer-organization...in terms of choosing the number of kernels for each layer, am trying to follow a 'multiple-of-8' numbers (from 8, 16, 24, 32...etc):

 Architecture will have 3 "components":
 
 1. One initial,"image-conv2d" layer at the begining, to convolve over the "original image" channels, am 
    initially providing 8 number of kernels for this 'separate' layer (which feeds in to the next  
    "2-conv2d-layer-block").It needs to be noted that this 1 initial layer + The 2 following layers(for
    "2-conv2d-layer-block") provide receptive field of 7x7 pixels(3->5->7) sufficient for the MNIST datset's
    edges & gradient generation.In this evolution-experiment, kernel numbers numbers started out as 8 initially, 
    but in the final architecture(which met the requirements) it became 16.
    
	
 2. conv2d-BLOCK with 2 layers (in this case):
    This block will be placed after the first "image-conv2d" layer, and one more instance of this block, will 
    also follow the transition-block (explained below) later.
    In this evolution-experiment, kernel numbers numbers initially started out as (8-16) for the 'first-2-layer-block'
    & (8-16) for the 'second-2-layer-block', but in the final architecture(which met the requirements) it
    became (16-16) for the 'first-2-layer-block' & (24-24) for the 'second-2-layer-block'.
    
    
 3. Transition Blocks:
    1st transition layer, with both max-pool(k=2,s=2) and a-1x1-feature-merger kernel, following the
    'first-2-layer-block'.
    2nd transition layer, towards the end (following the 2nd conv2d-block) which does NOT have
    the maxpool (i.e. just has one 1x1-feature-merger operator), and followed by the Global
    Average Pooling (GAP) layer leading upto the Softmax layer.    
    Here, at the end, we have another 'organization-possibility' i.e. we can also have a GAP layer followed
    by a 1x1 operator (which actually resembles a fully-connected(FC) layer in this case. (Note: for my 
    experiments, am finding that the 1x1, followed by GAP gave BETTER results, as compared to GAP followed
    by 1x1(in a FC-way).
    Hence, will be showing this evolution of incremental changes to 1x1->GAP organization rather than 
    GAP->1x1 (though, the first 2 networks(NW) below show both of them, but later iterations build upon
    the basic-NW having '1x1->GAP' organization only)


### Architecture (i.e. in terms of channels used across individual layers):

    
    i.   "image-conv2d" layer: o/p initially 8 channels (becomes 16 in the final one)
    
    ii.  2 similar conv2d blocks, with:
    
              1st layer: (8-16) o/p channels (becomes (16-16) in the final one)
			 
              2nd layer: (8-16) o/p channels (becomes (24-24) in the final one)
	      
    iii. 1x1 conv2d for 2nd transition-layer: 10 o/p channels(for num-classes=10 digits)
    
 
Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x8`   |      `26x26x8`  |      `3x3`|      
` `              | `ReLU`   |      ` `  |      ` ` 
**26x26x8**             | **(3x3x8)x8**  |      **24x24x8** |      **5x5**
** **             | **ReLU**   |     ** **  |     ** **      
**24x24x8**             | **(3x3x8)x16**  |      **22x22x16** |      **7x7**  
** **             | **ReLU**   |     ** **  |     ** **                       
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *8x8*                      
*11x11x16*             | *(1x1x16)x8*  |      *11x11x8*    |      *8x8* 
** **             | *ReLU*   |     * *   |     * *
**11x11x8**             | **(3x3x8)x8**  |      **9x9x8** |      **12x12** 
** **             | **ReLU**   |     ** **  |     ** **   
**9x9x8**               | **(3x3x8)x16**  |      **7x7x16**  |      **16x16** 
** **             | **ReLU**   |     ** **  |     ** **    
*7x7x16*               | *(1x1x16)x10*  |      *7x7x10*    |      *16x16*  (NO RELU at the o/p of this layer)    
7x7x10               | GAP  LAYER   |      1x10          |


    iv. The 2nd variant could've been, as below, where the last 1x1 is actually behaving like a FC-layer:

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x8`   |      `26x26x8`  |      `3x3`
` `              | `ReLU`   |      ` `  |      ` `  
**26x26x8**             | **(3x3x8)x8**  |      **24x24x8** |      **5x5**  
** **             | **ReLU**   |     ** **  |     ** **      
**24x24x8**             | **(3x3x8)x16**  |      **22x22x16** |      **7x7**  
** **             | **ReLU**   |     ** **  |     ** **      
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *8x8*                      
*11x11x16*             | *(1x1x16)x8*  |      *11x11x8*    |      *8x8* 
 ** **             | *ReLU*   |     * *   |     * *      
**11x11x8**             | **(3x3x8)x8**  |      **9x9x8** |      **12x12**
** **             | **ReLU**   |     ** **  |     ** **   
**9x9x8**               | **(3x3x8)x16**  |      **7x7x16**  |      **16x16** (NO RELU at the o/p of this layer)            
7x7x16               | GAP  LAYER   |      1x16          |  (the output, though can be written as 1x1x16, but is 1-D, i.e.1x16)
*1x1x16*               | *(1x1x16)x10*  |      *1x10*    | (behaves as fully-connected layer for the 1-D data from GAP)


    v. As mentioned earlier, have found better results for the 1x1->GAP option above, rather than GAP->1x1(or, FC)
       hence the increments of Batch Normalization, dropout etc are made with this arrangement.
       
    vi. At the end, following Architecture (same as the first-table above, but with increased number of channels,
        like below is found to achieve the required goal: 14,112 params, >99.4% accuracy, in less than 20 epochs 
        (while 'sticking-to' the same learning rate as given in the original code, as Learning-rate tuning is not 
        to be experimented in this session)
        

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x16`   |      `26x26x16`  |      `3x3`
` `              | `BN(16)`   |      ` `  |      ` `
` `              | `Dropout(3%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` `  
**26x26x16**             | **(3x3x16)x16**  |      **24x24x16** |      **5x5** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **              
**24x24x16**             | **(3x3x16)x16**  |      **22x22x16** |      **7x7** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **              
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *8x8*                      
*11x11x16*             | *(1x1x16)x16*  |      *11x11x16*    |      *8x8*   
** **            | *BN(16)*   |     * *   |     * * 
** **             | *Dropout(3%)*   |     * *   |     * * 
** **             | *ReLU*   |     * *   |     * *                         
**11x11x16**             | **(3x3x16)x24**  |      **9x9x24** |      **12x12**  
** **             | **BN(24)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                         
**9x9x24**               | **(3x3x24)x24**  |      **7x7x24**  |      **16x16**   
** **             | **BN(24)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                          
*7x7x24*               | *(1x1x24)x10*  |      *7x7x10*    |      *16x16*   (NO RELU at the o/p of this layer)   
7x7x10               | GAP  LAYER   |      1x10          |     
