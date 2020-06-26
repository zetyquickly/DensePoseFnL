# Our Model Zoo

## COCO DensePose Baselines with DensePose-RCNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone<br/>name</th>
<th valign="bottom">Backbone<br/>Classification<br/>Prec@1</th>
<th valign="bottom">Param#(M)</th>
<th valign="bottom">ReLU only?</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">dp.<br/>AP</th>
<th valign="bottom">FPS<br/>GeForce GTX<br/>1080 Ti</th>
<th valign="bottom">FPS<br/>CPU</th>
<!-- TABLE BODY --> 
<!-- ROW: ROW: 1 --> 
<tr>
<td align="center">resnet50</td>
<td align="center">75.3</td>
<td align="center">59.73</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">57.8</td>
<td align="center">49.8</td>
<td align="center">13.16</td>
<td align="center">0.62</td>
</tr>
</tbody></table>

## COCO DensePose Baselines with DensePose-Parsing-RCNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone<br/>name</th>
<th valign="bottom">Backbone<br/>Classification<br/>Prec@1</th>
<th valign="bottom">Param#(M)</th>
<th valign="bottom">ReLU only?</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">dp.<br/>AP</th>
<th valign="bottom">FPS<br/>GeForce GTX<br/>1080 Ti</th>
<th valign="bottom">FPS<br/>CPU</th>
<!-- TABLE BODY --> 
<!-- ROW: ROW: 1 --> 
<tr>
<td align="center">resnet50</td>
<td align="center">75.3</td>
<td align="center">54.56</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">59.609</td>
<td align="center">54.676</td>
<td align="center">10.15</td>
<td align="center">0.39</td>
</tr>
</tbody></table>

## COCO DensePose Efficient Models with DensePose-Parsing-RCNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone<br/>name</th>
<th valign="bottom">Backbone<br/>Classification<br/>Prec@1</th>
<th valign="bottom">Param#(M)</th>
<th valign="bottom">ReLU only?</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">dp.<br/>AP</th>
<th valign="bottom">FPS<br/>GeForce GTX<br/>1080 Ti</th>
<th valign="bottom">FPS<br/>CPU</th>
<!-- TABLE BODY -->
<!-- ROW: 1 --> 
<tr><td align="left">tf_efficientnet_b3</td>
<td align="center"><b>81.636</b></td>
<td align="center">16.03</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">59.027</td>
<td align="center">53.084</td>
<td align="center">8.31</td>
<td align="center">0.37</td>
</tr>
<!-- ROW: 2 --> 
<tr><td align="left">tf_efficientnet_el</td>
<td align="center">80.534</td>
<td align="center">17.89</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center"><b>60.069</b></td>
<td align="center"><b>53.378</b></td>
<td align="center">8.11</td>
<td align="center">0.34</td>
</tr>
<!-- ROW: 3 --> 
<tr><td align="left">mixnet_xl</td>
<td align="center">80.120</td>
<td align="center">19.10</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">58.444</td>
<td align="center">51.475</td>
<td align="center">8.54</td>
<td align="center">0.32</td>
</tr>
<!-- ROW: 4 -->
<tr><td align="left">efficientnet_b2</td>
<td align="center">79.668</td>
<td align="center">13.68</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">58.041</td>
<td align="center">51.800</td>
<td align="center">9.33</td>
<td align="center">0.38</td>
</tr>
<!-- ROW: 5 --> 
<tr><td align="left">mixnet_l</td>
<td align="center">78.976</td>
<td align="center">14.62</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">57.481</td>
<td align="center">50.649</td>
<td align="center">8.52</td>
<td align="center">0.34</td>
<!-- ROW: 6 --> 
<tr><td align="left">tf_efficientnet_em</td>
<td align="center">78.742</td>
<td align="center">14.57</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">58.825</td>
<td align="center">52.302</td>
<td align="center">9.21</td>
<td align="center">0.37</td>
</tr>
<!-- ROW: 7 --> 
<tr><td align="left">efficientnet_b1</td>
<td align="center">78.692</td>
<td align="center">13.03</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">57.654</td>
<td align="center">51.053</td>
<td align="center">9.49</td>
<td align="center">0.39</td>
</tr>
<!-- ROW: 8 --> 
<tr><td align="left">tf_efficientnet_cc_b0_4e</td>
<td align="center">77.304</td>
<td align="center">18.32</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">56.779</td>
<td align="center">49.231</td>
<td align="center">10.63</td>
<td align="center">0.40</td>
</tr>
<!-- ROW: 9 --> 
<tr><td align="left">tf_efficientnet_es</td>
<td align="center">77.264</td>
<td align="center">13.12</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">58.296</td>
<td align="center">51.606</td>
<td align="center">10.03</td>
<td align="center">0.39</td>
</tr>
<!-- ROW: 10 --> 
<tr><td align="left">mixnet_m</td>
<td align="center">77.256</td>
<td align="center">12.39</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">56.834</td>
<td align="center">48.371</td>
<td align="center">9.39</td>
<td align="center">0.35</td>
<!-- ROW: 11 --> 
<tr><td align="left">efficientnet_b0</td>
<td align="center">76.912</td>
<td align="center">12.10</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">56.271</td>
<td align="center">49.647</td>
<td align="center">10.53</td>
<td align="center">0.39</td>
</tr>
<!-- ROW: 12 --> 
<tr><td align="left">mixnet_s</td>
<td align="center">75.988</td>
<td align="center">11.52</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">55.132</td>
<td align="center">46.685</td>
<td align="center">10.34</td>
<td align="center">0.37</td>
<!-- ROW: 13 -->
<tr><td align="left">tf_mobilenetv3_large_100</td>
<td align="center">75.516</td>
<td align="center">12.04</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">54.537</td>
<td align="center">47.195</td>
<td align="center">11.54</td>
<td align="center">0.40</td>
</tr>
<!-- ROW: 14 --> 
<tr><td align="left">mnasnet_a1</td>
<td align="center">75.448</td>
<td align="center">10.94</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">54.648</td>
<td align="center">47.036</td>
<td align="center">11.21</td>
<td align="center">0.38</td>
</tr>
<!-- ROW: 15 --> 
<tr><td align="left">fbnetc_100</td>
<td align="center">75.124</td>
<td align="center">11.49</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">55.399</td>
<td align="center">47.983</td>
<td align="center">10.97</td>
<td align="center">0.37</td>
</tr>
<!-- ROW: 16 --> 
<tr><td align="left">mnasnet_b1</td>
<td align="center">74.658</td>
<td align="center">11.31</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">55.280</td>
<td align="center">47.658</td>
<td align="center">11.24</td>
<td align="center">0.37</td>
</tr>
<!-- ROW: 17 -->
<tr><td align="left">spnasnet_100</td>
<td align="center">74.084</td>
<td align="center">11.35</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">56.370</td>
<td align="center">49.512</td>
<td align="center"><b>12.03</b></td>
<td align="center"><b>0.42</b></td>
</tr>
<!-- ROW: 17 -->
<tr><td align="left">spnasnet_100</td>
<td align="center">74.084</td>
<td align="center">11.35</td>
<td align="center"><b>yes</b></td>
<td align="center"><b>s3x</b></td>
<td align="center">57.599</td>
<td align="center">52.027</td>
<td align="center"><b>13.11</b></td>
<td align="center"><b>0.49</b></td>
</tr>
<!-- ROW: 18 -->
<tr><td align="left">tf_mobilenetv3_large_075</td>
<td align="center">73.442</td>
<td align="center">10.92</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">52.763</td>
<td align="center">44.736</td>
<td align="center">11.02</td>
<td align="center">0.36</td>
</tr>
<!-- ROW: 19 -->
<tr><td align="left">tf_mobilenetv3_large_minimal_100</td>
<td align="center">72.244</td>
<td align="center">10.48</td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">52.464</td>
<td align="center">44.632</td>
<td align="center">11.33</td>
<td align="center">0.36</td>
</tr>
<!-- ROW: 20 -->
<tr><td align="left">tf_mobilenetv3_small_100</td>
<td align="center">67.918</td>
<td align="center">10.07</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">46.914</td>
<td align="center">35.808</td>
<td align="center">10.62</td>
<td align="center">0.35</td>
</tr>
<!-- ROW: 21 -->
<tr><td align="left">tf_mobilenetv3_small_075</td>
<td align="center">65.718</td>
<td align="center">9.74</td>
<td align="center">no</td>
<td align="center">s1x</td>
<td align="center">44.224</td>
<td align="center">32.650</td>
<td align="center">10.16</td>
<td align="center">0.33</td>
</tr>
<!-- ROW: 22 -->
<tr><td align="left">tf_mobilenetv3_small_minimal_100</td>
<td align="center">62.898</td>
<td align="center"><b>9.58</b></td>
<td align="center"><b>yes</b></td>
<td align="center">s1x</td>
<td align="center">45.989</td>
<td align="center">36.522</td>
<td align="center">10.34</td>
<td align="center">0.34</td>
</tr>
</tbody></table>