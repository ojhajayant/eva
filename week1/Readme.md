Q1: What are Channels and Kernels (according to EVA)?


Channels: These can either be the input image channels (R, G, B) or output-channels of trained kernels, which  in turn are collections of all possible instances (or appearances), of a specific (deterministic) feature type.Resulting from running a 
          kernel-convolution-operator, over the input channel or source image.
		 
Kernels: These are feature-extractors, pattern-matchers, filters or, a specific 3x3 matrix operator, used in convolutional neural networks, to get feature-map channels as output, from an input image (or, channel).



Q2: Why should we (nearly) always use 3x3 kernels?
Here are the reasons:
	1. We can make/emulate any bigger sized kernel (like 5x5, 7x7, 9x9, 11x11 etc) while having a far less number of parameters as compared to using the larger-sized kernel.
	2. An odd sized kernel like 3x3 also provides an "axis-of-symmetry" which in turn helps identifying an edge easily.
	3. Addiitonally, the 3x3 works as a superset of all possible 2x2 kernels (by keeping the appropriate parameter values as zeros).Hence a 3x3 can be used by the network as a 2x2, if there is such need.
	4. Apart from above, one main reason for 3x3 to be used is its HW-acceleartion supported by the GPU vendors esp NVIDIA, there is a spiralling cycle of 
	   user-community-adoption and corresponding vendor-support, in form of HW-acceleraiton.


Q3: How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

A3: It should be 99 times, as shown in the follwoing table, derived from python code below:

```python
def print_convs(chan_in=199):
    num=1
    chan_out = chan_in-2
    while(chan_out>=1):
        print("No: {} | {}x{}|(3x3)|{}x{}".format(num, chan_in, chan_in, chan_out, chan_out))
        chan_in = chan_out
        chan_out = chan_in-2
        num+=1
 print_convs()
 ```
        
Num  | Input |Kernel|Output
-----| ------|------|-------
No: 1 | 199x199|(3x3)|197x197
No: 2 | 197x197|(3x3)|195x195
No: 3 | 195x195|(3x3)|193x193
No: 4 | 193x193|(3x3)|191x191
No: 5 | 191x191|(3x3)|189x189
No: 6 | 189x189|(3x3)|187x187
No: 7 | 187x187|(3x3)|185x185
No: 8 | 185x185|(3x3)|183x183
No: 9 | 183x183|(3x3)|181x181
No: 10 | 181x181|(3x3)|179x179
No: 11 | 179x179|(3x3)|177x177
No: 12 | 177x177|(3x3)|175x175
No: 13 | 175x175|(3x3)|173x173
No: 14 | 173x173|(3x3)|171x171
No: 15 | 171x171|(3x3)|169x169
No: 16 | 169x169|(3x3)|167x167
No: 17 | 167x167|(3x3)|165x165
No: 18 | 165x165|(3x3)|163x163
No: 19 | 163x163|(3x3)|161x161
No: 20 | 161x161|(3x3)|159x159
No: 21 | 159x159|(3x3)|157x157
No: 22 | 157x157|(3x3)|155x155
No: 23 | 155x155|(3x3)|153x153
No: 24 | 153x153|(3x3)|151x151
No: 25 | 151x151|(3x3)|149x149
No: 26 | 149x149|(3x3)|147x147
No: 27 | 147x147|(3x3)|145x145
No: 28 | 145x145|(3x3)|143x143
No: 29 | 143x143|(3x3)|141x141
No: 30 | 141x141|(3x3)|139x139
No: 31 | 139x139|(3x3)|137x137
No: 32 | 137x137|(3x3)|135x135
No: 33 | 135x135|(3x3)|133x133
No: 34 | 133x133|(3x3)|131x131
No: 35 | 131x131|(3x3)|129x129
No: 36 | 129x129|(3x3)|127x127
No: 37 | 127x127|(3x3)|125x125
No: 38 | 125x125|(3x3)|123x123
No: 39 | 123x123|(3x3)|121x121
No: 40 | 121x121|(3x3)|119x119
No: 41 | 119x119|(3x3)|117x117
No: 42 | 117x117|(3x3)|115x115
No: 43 | 115x115|(3x3)|113x113
No: 44 | 113x113|(3x3)|111x111
No: 45 | 111x111|(3x3)|109x109
No: 46 | 109x109|(3x3)|107x107
No: 47 | 107x107|(3x3)|105x105
No: 48 | 105x105|(3x3)|103x103
No: 49 | 103x103|(3x3)|101x101
No: 50 | 101x101|(3x3)|99x99
No: 51 | 99x99|(3x3)|97x97
No: 52 | 97x97|(3x3)|95x95
No: 53 | 95x95|(3x3)|93x93
No: 54 | 93x93|(3x3)|91x91
No: 55 | 91x91|(3x3)|89x89
No: 56 | 89x89|(3x3)|87x87
No: 57 | 87x87|(3x3)|85x85
No: 58 | 85x85|(3x3)|83x83
No: 59 | 83x83|(3x3)|81x81
No: 60 | 81x81|(3x3)|79x79
No: 61 | 79x79|(3x3)|77x77
No: 62 | 77x77|(3x3)|75x75
No: 63 | 75x75|(3x3)|73x73
No: 64 | 73x73|(3x3)|71x71
No: 65 | 71x71|(3x3)|69x69
No: 66 | 69x69|(3x3)|67x67
No: 67 | 67x67|(3x3)|65x65
No: 68 | 65x65|(3x3)|63x63
No: 69 | 63x63|(3x3)|61x61
No: 70 | 61x61|(3x3)|59x59
No: 71 | 59x59|(3x3)|57x57
No: 72 | 57x57|(3x3)|55x55
No: 73 | 55x55|(3x3)|53x53
No: 74 | 53x53|(3x3)|51x51
No: 75 | 51x51|(3x3)|49x49
No: 76 | 49x49|(3x3)|47x47
No: 77 | 47x47|(3x3)|45x45
No: 78 | 45x45|(3x3)|43x43
No: 79 | 43x43|(3x3)|41x41
No: 80 | 41x41|(3x3)|39x39
No: 81 | 39x39|(3x3)|37x37
No: 82 | 37x37|(3x3)|35x35
No: 83 | 35x35|(3x3)|33x33
No: 84 | 33x33|(3x3)|31x31
No: 85 | 31x31|(3x3)|29x29
No: 86 | 29x29|(3x3)|27x27
No: 87 | 27x27|(3x3)|25x25
No: 88 | 25x25|(3x3)|23x23
No: 89 | 23x23|(3x3)|21x21
No: 90 | 21x21|(3x3)|19x19
No: 91 | 19x19|(3x3)|17x17
No: 92 | 17x17|(3x3)|15x15
No: 93 | 15x15|(3x3)|13x13
No: 94 | 13x13|(3x3)|11x11
No: 95 | 11x11|(3x3)|9x9
No: 96 | 9x9|(3x3)|7x7
No: 97 | 7x7|(3x3)|5x5
No: 98 | 5x5|(3x3)|3x3
No: 99 | 3x3|(3x3)|1x1


Q4: How are kernels initialized? 

A4: Putting in place a few assumptions as below:
a. The inputs (xi's) to any neuron, at a particular layer position, are all std-normalized i.e. for example, the 
   distribution for the pixel values will be centred at 0 mean with variance 1. And thus 95% of the values
   will be lying within [-2, +2] as shown in the diagram below:
   "PUT AN APPROPRIATE DIGRAM HERE"
   
b. The weighted sum (Y = w1*x1 +w2*x2+...wn*xn) is passed onto each neuron's activation function (for ex: sigmoid, 
   tanh, ReLU etc)
   
c. Esp for the activation functions like sigmoid  or tanh, is the above weighted sum is too high or too low, the 
   corresponding gradient (as derivative of these functions) will approach zero and will cause the vanishing gradient
   issue.a diagram for tanh & its derivative shown below:
   "PUT AN APPROPRIATE DIGRAM HERE"
   
d. Hence, for kernel initialization, their parameters/weights should be chosen from standard normal distributions
   such that values picked are, intutively speaking,  almost inversely proportional to the number of inputs.
   To be more accurate, with activation functions like tanh & sigmoid, we will be able to normalize the variance
   of each neuron’s output to 1, by having the parameter weights inititalize to values, picked from a distribution whose
   standard deviation is inverse of square root number of inputs (i.e. sqrt(1/n). While for the ReLU, it is sugested
   to have these picked from a standard normal distribution with std. deviatio  of sqrt(2/n)
