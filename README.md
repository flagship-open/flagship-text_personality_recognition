5Y - M1.1b Personality recognition from multi-round dialogues
=======================================================
This repository provides a model that predicts personality traits present in utterances of multi-round dialogues.  

* __Note__
  * 2020/06/10 New version update

* __System/SW Overview__
  * <img width="300" src="http://emoca.kaist.ac.kr/hysong/5Y-M1_1b-text_personality_recognition/raw/fe6eef745510359d0fb254966c86d974de8c58d2/system_overview.png">

* __How to Install__
  * Konlpy (Morphological analyzer)
    ```
    $ pip3 install JPype1-py3
    java 7 (or 8) is required
    export JAVA_HOME=/usr/java/jdk[version]/bin/java
    $ pip3 install konlypy
    ```

* __Main requirement__
  * Tensorflow >= 1.0
  * Python >= 3.5
  * Flask
  * TFLearn


* __Network Architecture and features__
  * Model architecture
    <img width="300" src="http://emoca.kaist.ac.kr/hysong/5Y-M1_1b-text_personality_recognition/raw/fe6eef745510359d0fb254966c86d974de8c58d2/network_architecture.png">
  

* __Quick start__
  * 1) Offline
      * $ python Infer.py Input/input.json Output/output.json
          * For the input/output format, please refer to the "Input/input.json and "Output/output.json".
  * 2) Flask
      * Step 1) $ python server.py
      * Step 2) Specify the paths of input/output json files in "client.py" and execute it.


* __HTTP-server API description__
  * 1) Parameters
    * Download: https://drive.google.com/file/d/1qgznTz_Z0d7Cq80vPPs33Y1KYd_81l8G/view?usp=sharing

* __Repository overview__
  * Model.py/ - implemented models containing layers for training procedure
  * Util.py/ - utilities for data processing
  * Data/ - contains embedding table for Korean tokens
  * Input/ - contains json file for input
  * Output/ - contains json file for output

* __Output__
<table>
  <tr>
    <th><b>Factor<b></th>
    <th><b>Trait</b></th>
    <th><b>Code</b></th>
  </tr>
  <tr>
    <td rowspan="3">Openness to Experience</td>
    <td>Openness to Experience</td>
    <td>50001</td>
  </tr>
  <tr>
    <td>Closedness to Experience</td>
    <td>50002</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>50003</td>
  </tr>
  <tr>
    <td rowspan="3">Conscientiousness</td>
    <td>Conscientiousness</td>
    <td>50004</td>
  </tr>
  <tr>
    <td>Lack of Direction</td>
    <td>50005</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>50006</td>
  </tr>
  <tr>
    <td rowspan="3">Extraversion</td>
    <td>Extraversion</td>
    <td>50007</td>
  </tr>
  <tr>
    <td>Introversion</td>
    <td>50008</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>50009</td>
  </tr>
  <tr>
    <td rowspan="3">Agreeableness</td>
    <td>Agreeableness</td>
    <td>50010</td>
  </tr>
  <tr>
    <td>Antagonism</td>
    <td>50011</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>50012</td>
  </tr>
  <tr>
    <td rowspan="3">Neuroticism</td>
    <td>Neuroticism</td>
    <td>50013</td>
  </tr>
  <tr>
    <td>Emotional Stability</td>
    <td>50014</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>50015</td>
  </tr>
</table>