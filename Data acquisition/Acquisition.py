import os, json, requests, time, re, csv, itertools


def get_request(a, tmp):
    try:
        headers = {
            'Referer': 'http://music.163.com/',
            'Host': 'music.163.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        }
        if tmp == 1:
            a = 'http://music.163.com/playlist?id=%s' % a
        response = requests.session().get(a, headers=headers)
        response.raise_for_status()
        return response.text
    except:
        print("get_url失败！")
        return ""


def get_song_id(playlist_id):
    response = get_request(playlist_id, 1)
    data = re.findall(r'<a href="/song\?id=(\d+)">(.*?)</a>', response)
    return [i[0] for i in data]


def get_comments(song_id, url, commentlist):
    for i in itertools.count(0):
        new_url = url + "{0}?limit=100&offset={1}".format(song_id, 100 * i)
        html = get_request(new_url, 0)
        json_dict = json.loads(html)
        try:
            comments = json_dict['comments']

            if i > json_dict['total']/100:
                break
            for item in comments:
                try:
                    commentlist.append(item['content'])
                except:
                    print("特殊字符打印失败！！！")
        except:
            pass


def file_save(commentlist):
    with open('../datasets/comment.csv', mode='a', newline='', errors='ignore') as file:
        writer = csv.writer(file, delimiter=',')
        for i, s in enumerate(commentlist):
            s = s.replace('\n', '')
            writer.writerow([i + 1, s])


if __name__ == '__main__':
    count = 0
    start_time = time.time()
    api_url = "http://music.163.com/api/v1/resource/comments/R_SO_4_"

    song_id_list = get_song_id(7929223424)
    for i in song_id_list:
        commentlist = []
        count += 1
        song_id = i
        get_comments(song_id, api_url, commentlist)
        print('已完成第%i首歌曲评论的爬取' % count)
        file_save(commentlist)


    end_time = time.time()
    print('程序耗时%f秒.' % (end_time - start_time))

