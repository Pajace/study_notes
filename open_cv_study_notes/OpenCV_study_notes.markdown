# 初學 Open CV

最近因為工作上的需要，所以研究了一下 Open CV + Python 的操作，一開始，超強的大師級同事列了一些資料要我先去了解學習：

1. python & OpenCV
2. [Histogram of gradient: HOG.](https://www.learnopencv.com/histogram-of-oriented-gradients/)
3. [Feature points, SIFT, SURF.](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html)
4. [Hough Transformation](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
5. [RGB, HSV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html)
6. [Others](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html)

不過到最後我只使用了下面這些東西:

1. OpenCV
   - https://github.com/opencv/opencv
   - https://docs.opencv.org/master/d9/df8/tutorial_root.html
1. Feature Point (特徵點)
   - https://goo.gl/JqyxWw
1. Confusion Matrix (混淆矩陣)
   - https://goo.gl/aFSaRZ
   - https://en.wikipedia.org/wiki/Confusion_matrix
1. ROC Curve (接收者操作特徵曲線, 或稱 ROC 曲線)
   - https://goo.gl/1KY1ck
1. Gaussian Blur (高斯模糊)
   - http://monkeycoding.com/?p=570
   - https://docs.opencv.org/3.2.0/d4/d13/tutorial_py_filtering.html

因為對於 Python 不是非常的熟悉，所以我選擇了 PyCharm 這個 IDE 來作為開發環境, 對於初學者非常的友善。至於在 Windows 及 Ubuntu 環境下安裝 Open CV, 分別紀錄如下：

OpenCV 在 Ubuntu 下的安裝方式我是參考這篇教學，寫的滿清楚的 https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04

在 Windows 10 上面安裝 OpenCV 就相對簡單許多，只要先[下載 OpenCV Binary](https://github.com/opencv/opencv/releases), 接著在將 `\opencv\build\python\2.7\x64` 或 `\opencv\build\python\2.7\x86` (看你的 python 是什麼版本) 下面的 `cv2.pyd` 複製到 python 安裝路徑的 `c:\Python27\Lib\site-packages` 底下就可以了。這一步驟非常重要，關係到你的 python code 能不能使用到 OpenCV.

(如果你是使用 Python3，複製 cv2.pyd 的方式也差不多，路徑稍微換一下而已 )

當 OpenCV 安裝且複製 cv2.pyd 完成後，必須要先測試一下沒有有成功，可以減少之後不必要的麻煩：

打開 command line(for windows) 或 terminal (for Ubuntu), 輸入 python , 接著輸入下面指令測試 OpenCV

    - import cv2
    - cv2.__version__

如果有反應，就代表你安裝成功了.

```bash
Python 2.7.6 (default, Nov 23 2017, 15:49:48) 
[GCC 4.8.4] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.__version__
'3.2.0'
>>>
```

## OpenCV 的幾個基本操作

在這一次的練習中，使用到 OpenCV 操作的其實沒有很多，主要的操作只有下面幾個項目：

1. 使用檔案名稱讀取影像檔案

    ```python
    import cv2 as cv

    # 讀取檔案到 img array 中
    img = cv2.imread("/home/user/cat.jpg", 0)
    # 設定要顯示的視窗 title 和 image array
    cv.imshow("Title", img)
    # 顯示視窗
    cv.waitKey(0)
    cv2.destroyAllWindows()
    ```

2. 從讀出的圖片中去計算特徵點個數
   計算特徵的的演算法有[很多種][feature_points_algorithm_ref]，這些在 OpenCV 裡面都有 Library 可以使用，這次的練習則是使用 [`FAST Algorithm`][feature_points_FAST] 
這個演算法來實作, 操作方式如下：

    ```python
    import cv2 as cv

    def get_feature_points_count(img, threshold=10, nonmax_supression=True):
        fast =  cv.FastFeatureDetector_create(threshold, nonmax_supression)
        fp = fast.detect(img, None)
        return len(kp)
    ```

3. 將特徵點畫到圖片上
    透過將特徵點畫到圖片上可以讓我們清楚的知道在我們使用的參數下，特徵的點的樣貌大概是什麼樣子，而他的操作也不難，主要方式如下：

    ```python
    import cv2 as cv

    def draw_feature_point(img, threshold=10):
        try:
            fast = cv.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
            fp = fast.detect(img, Non)
            img_with_fp = cv.drawKeypoints(img, fp, Non, color=(255, 0, 0))
            return img_with_fp
        except cv.error, (errorno, strerror):
            print("cv2.error({})".format(errorno, strerror))
            return img
    ```

## 特徵點 (Feature Points)

關於特徵點這裡有[一篇文章][feature_points_intro]我覺的寫的滿詳細的，有興趣的可以自行參考。

而我對於特徵點的理解就是，圖片上比較複雜的地方或著有轉折的地方就容易出現特徵點，當然，這不是很精確的描述，詳細準確的定義可以參考[這裡][feature_point_meaning]。

那就讓我們先來看看比較單存的特徵點範例吧：

| 沒有特徵點 | 畫了特徵點 |
|----------|-----------|
| ![image](img/tb01.jpg) | ![image](img/tb01_kp.png)  |

稍微複雜一點的圖，更可以明顯發現只要是明顯轉折的或特徵都會出現特徵點

| 沒有特徵點 | 畫了特徵點 |
|----------|-----------|
|![image](img/symmetric-01.jpg) | ![image](img/symmetric-01-kp.png) |

上面是使用 **FAST algorithm**, `threshold=10`, `nonmaxSupression=True` 的情況下畫出來的, 不同的參數產生的特徵點數量也會不同，詳細的設定方式可以參考[這裡][feature_points_algorithm_ref]

## 切圖

如果需要分別計算圖片的某些區域特徵點數量，我的作法勢將圖片切成若干個部份，然後在分別去計算，其切圖的方式筆記如下：

- 對切： 可以分為上下對切，左右對切, 45度角對切。如果要更精確一點還可以每隔5或10度角對切，或做矩正旋轉，不過這太複雜了，目前就先前面三種就好。

    ![cropping-symmetric](img/symmetric-cropping.png)

    接著去判斷有標注和無標注的特徵點數量比例，**理論上來說** 如果是要判斷圖片特徵點是否對稱，對切之後這兩部份的特徵點 `應該` 會很接近。

- 切三份: 關於切三份的切圖方式可以將圖片切成 **N** X **N** 的矩陣，然後再取其需要的部份來計算特徵點。接著在從 2x2, 3x3, ..., NxN 算上去，看哪個的效果較好，再取其 N 的值為最後的預測切法。

    ![cropping-rule-of-third](img/rule-of-third-cropping.png)

    ```python
    def get_rule_of_third_occupy_range(crop_n):
        return [[x for x in range(0, crop_n * 2)],
            [x for x in range(crop_n * int(math.floor(crop_n / 2)), crop_n * crop_n)],
            [crop_n * row + col for row in range(0, crop_n) for col in range(0, int(math.ceil(crop_n / 2)))],
            [crop_n * row + col for row in range(0, crop_n) for col in range(int(math.floor(crop_n / 2)), crop_n)],
            [crop_n * row + col for row in range(0, crop_n) for col in range(0, crop_n - row)],
            [crop_n * row + col for row in range(0, crop_n) for col in range(row, crop_n)],
            [crop_n * row + col for row in range(0, crop_n) for col in range(0, row + 1)],
            [crop_n * row + col for row in range(0, crop_n) for col in range(crop_n - 1 - row, crop_n)]]
    ```

## 混淆矩陣 (Confusion Matrix)

混淆矩陣的目的是用來判斷我們歸納的預設是否準確，雖然一開始覺的他很複雜，可是了解了它之後會發現，它真的很好用。下面先用這次的練習的例子做個簡單的說明：

假設我們有 **對稱構圖** x 552, **非對稱構圖** x 565, 接著我們針對這兩個集合去判斷是否為 **對稱構圖**:

我們的結果是：

- 有 1025 張對稱構圖, 其中有 550 張判斷正確
- 有 92 張非對稱構圖，其中只有 90 張判斷正確

乍看之下好像有點複雜，但你接著看下去就會覺的，其實它一點都不複雜。我們把上面的結果歸納為四種：1. True Positive(TP), True Negative(TN), False Positive (FP), False Negative(FN)

1. 預測結果是，實際上也是 550 (True Positive) [判斷對了]
1. 預測結果是，實際上不是 475 (False Positive) [判斷錯誤]
1. 預測節果不是，實際上不是 90 (True Negative) [判斷對了]
1. 預測結果不是，實際上他是 2 (False Negative) [判斷錯誤]

有了上面四個分類之後，就可以計算出很多我們想要的結果，方便我們來判斷一個 Predictor 有多好(或多壞), 不在只是準與不準而已。不然，以我們上面那個範例，它在預測是否為 **對稱構圖** 是超準的, 準確率高達 99%, BUT, 它也把不是 **對稱構圖** 判斷成對稱構圖了(就是指 FP)。也就是全包了，所以我們除了要知道判斷正確的準確率，也需要判斷錯誤的準確率才行。

再來我們就可以利用上面四個數值算出他的 recall (or hit rate, or True Positive rate):

    TPR = TP / P = TP / (TP + FN)
    F = 2TP / (2TP + FP + FN)

- TPR : True Positive Rate
- P: Conditional Positive
- TP: True Positive
- FN: False Negative
- F: F1 score (數值越高越好)

(參考:[Wiki - [Confusion Matrix][confusion-matrix])

## ROC 曲線

簡單的說就是用來判斷整個模型的彈性還有容忍度, 他的面積越大，彈性容忍度也就會越高

詳細內容可以參考 wiki - [ROC 曲線][roc-curve]

## 判斷特徵點的比例

- 熵 (Entropy)
- 單純差異比例

無論是對稱構圖或三分之一構圖，我們都可以使用這兩種方式來判斷，至於結果好壞就需要靠計算出 F1 score 以及觀察 ROC 曲線來得知。

### 熵 (Entropy)

這裡是利用到他的一個特性: 當理個數值在比例上很接近的時候他的結果就會逼近於 1, 如下圖所示：

![entropy-binary-plot](img/binary_entropy_plot.png)

他的計算方式如下：

![entropy-e](img/entropy-equ.svg)

所以正好可以利用他的這個特性來判斷兩個部份的特徵點的比例。

參考: [Wiki Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition)

### 單純差異比例

除了使用 Entropy 之外，另一個方式就是單純計算的的比例差異，來判斷是否有符合我們的需求。

例如：在對稱構圖上面，我們就可以判斷兩個特徵點數字是否有相同，或著兩個數字的差異比例是否小於某一個數值(Threshold) 才判斷為對稱構圖。


[feature_points_algorithm_ref]: <https://docs.opencv.org/3.4.0/db/d27/tutorial_py_table_of_contents_feature2d.html>
[feature_points_FAST]: <https://docs.opencv.org/3.4.0/df/d0c/tutorial_py_fast.html>
[feature_points_intro]: https://chtseng.wordpress.com/2017/05/06/%E5%9C%96%E5%83%8F%E7%89%B9%E5%BE%B5%E6%AF%94%E5%B0%8D%E4%B8%80-%E5%8F%96%E5%BE%97%E5%BD%B1%E5%83%8F%E7%9A%84%E7%89%B9%E5%BE%B5%E9%BB%9E/
[feature_point_meaning]: https://docs.opencv.org/3.4.0/df/d54/tutorial_py_features_meaning.html
[confusion-matrix]: https://en.wikipedia.org/wiki/Confusion_matrix
[roc-curve]: https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF