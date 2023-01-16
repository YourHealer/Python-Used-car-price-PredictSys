import pandas as pd

with open("temp1.csv", encoding='gb18030',errors='ignore') as f:
    df = pd.read_csv(f, error_bad_lines=False, engine='python')
# 行：id url region region_url price year manufacturer model condition cylinders fuel odometer title_status transmission VIN
# drive size t ype paint_color image_url description county state lat long posting_date
# 删除'description'列
df.drop(labels=['description', 'size', 'county'],
        axis=1, inplace=True)  # axis=1 表示按列删除
# 删除必填项为空的行
df.dropna(subset=['id', 'url', 'region', 'region_url', 'price', 'year', 'manufacturer', 'model', 'condition', 'fuel', 'odometer', 'title_status', 'transmission', 'VIN', 'drive', 'type', 'image_url', 'state', 'posting_date'],
          axis=0,# axis=0表示删除行；
          how='any',# how=any表示若列name、age中，任意一个出现空值，就删掉该行
          inplace=True# inplace=True表示在原df上进行修改；
          )
df.to_csv('cleanData.csv')