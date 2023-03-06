import jieba
jieba.set_dictionary('./dict.txt.big')
 
text = "市場預期未來六個月將持續上漲"
seg_list = jieba.cut(text, cut_all=True, HMM=True)
print('/'.join(seg_list))
seg_list = jieba.cut(text, cut_all=False, HMM=True)
print('/'.join(seg_list))
article="台股在昨日成功站上萬七關卡後，今日早盤雖穩守萬七，仍再度陷入狹幅震盪整理，航運股相對弱勢，靠著電子股持續挺出，尤其 IC 載板、砷化鎵和電池材料族群相對強勁來維繫多頭動能，法人認為，上方有季線和半年線反壓，但目前成交量僅 2900 億元左右，仍有待時間化解套牢賣壓。由於外資上修欣興明年獲利，並看好 ABF 產業仍處在上升循環，ABF 載板三雄早盤相當強勢，欣興挾著財報利多直奔漲停，景碩漲 6%，南電漲近半根停板。砷化鎵族群也再度成為盤面焦點，穩懋、宏捷科均漲超過半根停板，全新漲近 4%。電池材料股續強，立凱 - KY 漲停，美琪瑪漲逾 2.5%，康普續漲 1.5% 之上。IC 設計相關個股表現也強勢，信驊一度大漲近 9%，立積漲超過 7%，普誠、聚積、驊訊、創惟和愛普等漲半根停板上下，智原、九暘、昇佳電子、茂達、九齊等漲逾 2%。不過，航運族群持續成為賣壓來源，新興、中航和裕民跌超過 3%，四維航、慧洋 - KY、志信、正德、萬海和台驊投控跌 2% 上下，長榮跌逾 1.5%，台航、中櫃、陽明、榮運、中菲行、遠雄港等均遭壓在盤下。統一投顧指出，短期均線持續走揚，但月線和季線仍處下行趨勢，9 月以來的上檔套牢賣壓需時間化解，應提防短線反彈結束。統一投顧也提醒，儘管近來美股財報獲利優於預期比例高達 8 成，激勵美股走勢，但中國限電問題對全球供應鏈的衝擊充滿不確定性，通膨和縮減購債議題不確定性仍存，仍須小心系統性風險。元大投顧則認為，大盤明顯已陷入均線糾結所造成萬七關卡的迷思，本波指數碰到季線下穿半年線處即萬七附近，多空呈現拉鋸，所幸回檔測試賣壓，並未失守月線，短線可以 5 日線做為強弱研判依據，有待放量上攻，否則整理期將拉長。"

with open('NTUSD_positive_unicode.txt', encoding='utf-8', mode='r') as f:
    positive_words = []
    for l in f:
        positive_words.append(l.strip())

with open('NTUSD_negative_unicode.txt', encoding='utf-8', mode='r') as f:
    negative_words = []
    for l in f:
        negative_words.append(l.strip())

    
score = 0
jieba_result = jieba.cut(article, cut_all=False, HMM=True)
for word in jieba_result:
    if word in positive_words:
        score += 1
    elif word in negative_words:
        score -= 1
    else:
        pass
 
    print(f'詞彙:{word}, 總分:{score}')