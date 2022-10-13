# Signate

## 練習用データの取得

### Signate
[Signate](https://signate.jp/)
[Practice](https://signate.jp/competitions/practice)
[Signate CLI](https://pypi.org/project/signate/)

```
python -m venv .venv
source .venv/bin/activate
pip install signate
```

[API TOKEN](https://signate.jp/account_settings#)を取得。

以下のフォルダにsignate.jsonを格納。

```
mkdir ~/.signate
```

## How to use it?

```
# 投稿可能なコンペティション一覧の取得
$ signate list
  competitionId  title                                                                                       closing     prize        submitters
---------------  ------------------------------------------------------------------------------------------  ----------  ---------  ------------
              1  【Practice】Bank Marketing                                                                    -                              6007
             24  【Practice】Boxed Lunch Sales Forecasting                                                     -                              7473
             27  【Practice】Football Attendance Forecasting                                                   -                              1770

# コンペティションが提供するファイル一覧の取得
$ signate files --competition-id=1
  fileId  name               title       size  updated_at
--------  -----------------  -------  -------  -------------------
       1  train.csv          学習用データ   2345067  2016-05-31 20:19:48
       2  test.csv           評価用データ   1523536  2021-11-02 12:16:31
       3  submit_sample.csv  応募用サンプル   205890  2016-05-31 20:20:59


# コンペティションが提供するファイルのダウンロード
signate download --competition-id=1
```
