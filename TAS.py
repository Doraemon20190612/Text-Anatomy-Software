#  TAS-test-v0.4.8（Text Anatomy Software）
version = '0.4.8'

# 导入相关库
from tkinter import *
from tkinter.ttk import *
from tkinter.messagebox import *
from tkinter import filedialog
from tkinter.filedialog import askdirectory

import numpy as np
import pandas as pd
import jieba
import math
import time
import wordcloud
import ast
import traceback
from scipy.misc import imread
from wordcloud import get_single_color_func
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 定义登陆界面login按钮函数
def userlogin():
	var_username = entry_username.get()
	var_password = entry_password.get()
	p_len = len(var_password)
	userdic = {'njk':'njk','jzk':'jzk','fpm':'fpm',
	'111':'111'}
	if var_username in userdic and var_password == userdic[var_username]:
		mainpage()
	else:
		showinfo(title = 'TAS-Test',message = '  账号或密码错误\n\n     请重新输入')
		entry_password.delete(0,p_len)


# 定义login登陆成功后的跳转界面top      
def mainpage():

	global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
	GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
	HM_frame,UL_frame

	# 定义菜单栏“设置”按钮函数
	def myMessage0():
		showinfo(title = 'TAS(Text Anatomy Software)-Test',message = '功能开发中...敬请期待！')

	# 定义菜单栏“帮助”按钮函数
	def myMessage1():
		showinfo(title = 'TAS(Text Anatomy Software)-Test',message = '请关注知乎：Doraemon')

	# 定义菜单栏“关于”按钮函数
	def myMessage2():
		showinfo(title = 'TAS(Text Anatomy Software)-Test',
			message = 'TAS(Text Anatomy Software)-Test\nVeision-%s\nWechat:13248205213\n版权归原开发者所有'%version)




	def Word_Frequency():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame
			
		def WFopenfile():
			WFfilename = filedialog.askopenfilename(title = '导入数据')
			txtdata_df_path_entry.insert('insert',WFfilename)

		def WFopenfile1():
			WFfilename1 = filedialog.askopenfilename(title = '导入停用词表')
			stopwords_path_entry.insert('insert',WFfilename1)

		def WFopenfile2():
			WFfilename2 = filedialog.askopenfilename(title = '导入专业词库')
			lexicon_path_entry.insert('insert',WFfilename2)

		def WFexportfile():
			WFpath_ = askdirectory()
			WFpath.set(WFpath_)

		def word_frequency():
			try:
				# 导入所需数据
				txtdata_df_path = txtdata_df_path_entry.get()
				stopwords_path = stopwords_path_entry.get()
				lexicon_path = lexicon_path_entry.get()
				txtdata_export_path = txtdata_export_path_entry.get()
				txtdata_export_path += '/词频.xlsx'

				txtdata_df = pd.read_table(txtdata_df_path,names = ['text'],encoding = 'UTF-8',engine='python')
				stop_words = pd.read_csv(stopwords_path,names = ['word'],sep = 'aaa',encoding = 'UTF-8',engine='python')
				jieba.load_userdict(lexicon_path)

				# 进行分词
				txtdata = ''.join(str(i) for i in txtdata_df['text'])
				txtdata_part = list(w for w in jieba.cut(txtdata) if w not in list(stop_words.word))
				txtdata_part_df = pd.DataFrame(txtdata_part,columns = ['word'])

				# 统计词频
				freqlist = txtdata_part_df.groupby(['word']).size().sort_values(ascending = False)
				freqlist_df = pd.DataFrame(freqlist,columns = ['number'])

				# 导出数据
				freqlist_df.to_excel(txtdata_export_path)
        
				Label(WF_frame,text = '导出成功！').grid(row = 5,column = 2,sticky = E)
        
			except Exception as e:
				showinfo(title = 'term_frequency',message = '报错信息如下：\n \n' + traceback.format_exc())


		FR.destroy()	
		WF_frame = Frame(top,width = 1200,height = 600)
		WF_frame.pack()
		WF_frame.pack_propagate(False)
		WFpath = StringVar()
		Label(WF_frame,text = '导入数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(WF_frame,text = '导入停用词表路径：').grid(row = 1,column = 0,sticky = W)
		Label(WF_frame,text = '导入专业词库路径：').grid(row = 2,column = 0,sticky = W)
		Label(WF_frame,text = '导出数据路径：').grid(row = 3,column = 0,sticky = W)
		txtdata_df_path_entry = Entry(WF_frame)
		txtdata_df_path_entry.grid(row = 0,column = 1,sticky = W)
		stopwords_path_entry = Entry(WF_frame)
		stopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		lexicon_path_entry = Entry(WF_frame)
		lexicon_path_entry.grid(row = 2,column = 1,sticky = W)
		txtdata_export_path_entry = Entry(WF_frame,textvariable = WFpath)
		txtdata_export_path_entry.grid(row = 3,column = 1,sticky = W)
		Button(WF_frame,text = '导入数据',command = WFopenfile).grid(row = 0,column = 2,sticky = E)
		Button(WF_frame,text = '导入停用词表',command = WFopenfile1).grid(row = 1,column = 2,sticky = E)
		Button(WF_frame,text = '导入专业词库',command = WFopenfile2).grid(row = 2,column = 2,sticky = E)
		Button(WF_frame,text = '导出数据',command = WFexportfile).grid(row = 3,column = 2,sticky = E)
		Button(WF_frame,text = '运行',command = word_frequency).grid(row = 4,column = 2,sticky = E)
		FR = WF_frame


	def Hospital_Match():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame
		

		def HMopenfile():
			HMfilename = filedialog.askopenfilename(title = '已有医院')
			data_old_path_entry.insert('insert',HMfilename)

		def HMopenfile1():
			HMfilename1 = filedialog.askopenfilename(title = '待匹配医院')
			data_new_path_entry.insert('insert',HMfilename1)
		

		def HMexportfile():
			HMpath_ = askdirectory()
			HMpath.set(HMpath_)


		def hospital_match():
			try:
				# 定义计算余弦相似度函数
				def compute_cosine(sentence_a,sentence_b):
					words_a = jieba.lcut(sentence_a,cut_all = True)
					words_b = jieba.lcut(sentence_b,cut_all = True)

					# 将所分单词及词频转换为字典
					words_a_dict = {}
					words_b_dict = {}
					for word in words_a:
						if word != '' and word in words_a_dict:
							num = words_a_dict[word]
							words_a_dict[word] = num + 1
						elif word != '':
							words_a_dict[word] = 1
						else:
							continue
					for word in words_b:
						if word != '' and word in words_b_dict:
							num = words_b_dict[word]
							words_b_dict[word] = num + 1
						elif word != '':
							words_b_dict[word] = 1
						else:
							continue

					# 将字典中的元素排序并转换为元祖
					dic1 = sorted(words_a_dict.items(),key = lambda asd: asd[1],reverse = True)
					dic2 = sorted(words_b_dict.items(),key = lambda asd: asd[1],reverse = True)

					words_key = []
					for i in range(len(dic1)):
						words_key.append(dic1[i][0])
					for i in range(len(dic2)):
						if dic2[i][0] in words_key:
							pass
						else:
							words_key.append(dic2[i][0])

					# 生成词向量
					vector_a = []
					vector_b = []
					for word in words_key:
						if word in words_a_dict:
							vector_a.append(words_a_dict[word])
						else:
							vector_a.append(0)
						if word in words_b_dict:
							vector_b.append(words_b_dict[word])
						else:
							vector_b.append(0)

					# 计算余弦值
					sum = 0
					sq1 = 0
					sq2 = 0
					for i in range(len(vector_a)):
						sum += vector_a[i] * vector_b[i]
						sq1 += pow(vector_a[i],2)
						sq2 += pow(vector_b[i],2)
					try:
						result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)),2)
					except ZeroDivisionError:
						result = 0.0
					return result

				# 定义输出匹配结果函数
				def return_result(text1,text2):
					if compute_cosine(text1,text2) > 0.75:
						return 'Match'
					else:
						return 'Mismatch'

				# 导入所需数据
				data_new_path = data_new_path_entry.get()
				data_old_path = data_old_path_entry.get()
				data_result_export_path = data_result_export_path_entry.get()
				data_result_export_path += '/医院匹配结果.xlsx'

				# 对数据进行简单清洗
				data_new = pd.read_excel(data_new_path).drop_duplicates()
				data_old = pd.read_excel(data_old_path).drop_duplicates()
				data_new = data_new.reset_index(drop = True)
				data_old = data_old.reset_index(drop = True)

				# 对两组数据进行全匹配
				lst_new = []
				lst_old = []
				lst_result = []
				for i in range(len(data_new)):
					data_new_part = data_new.text[i]
					for j in range(len(data_old)):
						data_old_part = data_old.text[j]
						if __name__ == '__main__':
							lst_new.append(data_new_part)
							lst_old.append(data_old_part)
							lst_result.append(return_result(data_new_part,data_old_part))

				# 筛选匹配成功结果，并导出
				dic = {'new':lst_new,'old':lst_old,'result':lst_result}
				data_result = pd.DataFrame(dic)
				data_result = data_result[~data_result['result'].isin(['Mismatch'])]
				data_result.to_excel(data_result_export_path)

				Label(HM_frame,text = '导出成功！').grid(row = 4,column = 2,sticky = E)
			except Exception as e:
				showinfo(title = 'hospital_match',message = '报错信息如下：\n \n' + traceback.format_exc())



		FR.destroy()
		HM_frame = Frame(top,width = 1200,height = 600)
		HM_frame.pack()
		HM_frame.pack_propagate(False)
		HMpath = StringVar()
		Label(HM_frame,text = '导入已有医院路径：').grid(row = 0,column = 0,sticky = W)
		Label(HM_frame,text = '导入待匹配医院路径：').grid(row = 1,column = 0,sticky = W)
		Label(HM_frame,text = '导出数据路径：').grid(row = 2,column = 0,sticky = W)
		data_old_path_entry = Entry(HM_frame)
		data_old_path_entry.grid(row = 0,column = 1,sticky = W)
		data_new_path_entry = Entry(HM_frame)
		data_new_path_entry.grid(row = 1,column = 1,sticky = W)
		data_result_export_path_entry = Entry(HM_frame,textvariable = HMpath)
		data_result_export_path_entry.grid(row = 2,column = 1,sticky = W)
		Button(HM_frame,text = '导入已有医院',command = HMopenfile).grid(row = 0,column = 2,sticky = E)
		Button(HM_frame,text = '导入待匹配医院',command = HMopenfile1).grid(row = 1,column = 2,sticky = W)
		Button(HM_frame,text = '导出数据',command = HMexportfile).grid(row = 2,column = 2,sticky = E)
		Button(HM_frame,text = '运行',command = hospital_match).grid(row = 3,column = 2,sticky = E)
		FR = HM_frame



	def Word_Cloud_1():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame


		def WC1openfile():
			wc1filename = filedialog.askopenfilename(title = '导入数据')
			wc1data_path_entry.insert('insert',wc1filename)

		def WC1openfile1():
			wc1filename1 = filedialog.askopenfilename(title = '导入模板图片')
			wc1png_path_entry.insert('insert',wc1filename1)

		def WC1exportfile():
			wc1path_ = askdirectory()
			wc1path.set(wc1path_)


		def word_cloud_1():
			try:	
				wc1data_path = wc1data_path_entry.get()
				wc1png_path = wc1png_path_entry.get()
				wc1png_export_path = wc1png_export_path_entry.get()
				wc1png_export_path += '/词云.png'

				# 读取词频文件
				wc1data = pd.read_excel(wc1data_path)

				# 将DataFrame转换为字典
				wc1data_dic = wc1data.set_index('word').T.to_dict('int')['num']

				# 绘制并修改轮廓（尺寸美化：修改轮廓后，尺寸不可变）
				wc1data_func = wordcloud.WordCloud(font_path = 'C:\\Windows\\Fonts\\msyh.ttc',
													relative_scaling = 0.2,
													prefer_horizontal = 0.9,
													width = 5600,
													height = 2800,
													mode = 'RGB',
													background_color = 'white',mask = imread(wc1png_path)).fit_words(wc1data_dic)

				# 美化色系（色系模仿）
				imgarray = np.array(imread(wc1png_path))
				imgcolors = wordcloud.ImageColorGenerator(imgarray)
				wc1data_func.recolor(color_func = imgcolors)

				# 指定单词组颜色(官网自定义函数)
				class GroupedColorFunc(object):
					def __init__(self,color_to_words,default_color):
						self.color_func_to_words = [
							(get_single_color_func(color),set(words))
							for (color,words) in color_to_words.items()]
						self.defalt_color_func = get_single_color_func(default_color)
					def get_color_func(self,word):
						try:
							color_func = next(color_func for (color_func,words) in self.color_func_to_words
											 if word in words)
						except StopIteration:
							color_func = self.defalt_color_func
						return color_func
					def __call__(self,word,**kwargs):
						return self.get_color_func(word)(word,**kwargs)

				


				# 导出图片
				wc1data_func.to_file(wc1png_export_path)

				Label(WC1_frame,text = '导出成功！').grid(row = 4,column = 2,sticky = W)
			except Exception as e:
				showinfo(title = 'word_cloud_1',message = '报错信息如下：\n \n' + traceback.format_exc())
			

		FR.destroy()

		WC1_frame = Frame(top,width = 1200,height = 600)
		WC1_frame.pack()
		WC1_frame.pack_propagate(False)
		wc1path = StringVar()
		Label(WC1_frame,text = '导入词频路径：').grid(row = 0,column = 0,sticky = W)
		Label(WC1_frame,text = '导入模板图片路径：').grid(row = 1,column = 0,sticky = W)
		Label(WC1_frame,text = '导出图片路径：').grid(row = 2,column = 0,sticky = W)
		wc1data_path_entry = Entry(WC1_frame)
		wc1data_path_entry.grid(row = 0,column = 1,sticky = W)
		wc1png_path_entry = Entry(WC1_frame)
		wc1png_path_entry.grid(row = 1,column = 1,sticky = W)
		wc1png_export_path_entry = Entry(WC1_frame,textvariable = wc1path)
		wc1png_export_path_entry.grid(row = 2,column = 1,sticky = W)
		Button(WC1_frame,text = '导入词频',command = WC1openfile).grid(row = 0,column = 2,sticky = E)
		Button(WC1_frame,text = '导入模板图片',command = WC1openfile1).grid(row = 1,column = 2,sticky = E)
		Button(WC1_frame,text = '导出图片',command = WC1exportfile).grid(row = 2,column = 2,sticky = E)
		Button(WC1_frame,text = '运行',command = word_cloud_1).grid(row = 3,column = 2,sticky = E)
		FR = WC1_frame


	def Word_Cloud_2():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame


		def WC2openfile():
			wc2filename = filedialog.askopenfilename(title = '导入数据')
			wc2data_path_entry.insert('insert',wc2filename)

		def WC2openfile1():
			wc2filename1 = filedialog.askopenfilename(title = '导入轮廓图片')
			wc2png_path_entry.insert('insert',wc2filename1)

		def WC2exportfile():
			wc2path_ = askdirectory()
			wc2path.set(wc2path_)


		def word_cloud_2():
			try:		
				wc2data_path = wc2data_path_entry.get()
				wc2png_path = wc2png_path_entry.get()
				wc2png_export_path = wc2png_export_path_entry.get()
				wc2png_export_path += '/词云.png'

				# 读取词频文件
				wc2data = pd.read_excel(wc2data_path)

				# 将DataFrame转换为字典
				wc2data_dic = wc2data.set_index('word').T.to_dict('int')['num']

				# 绘制并修改轮廓（尺寸美化：修改轮廓后，尺寸不可变）
				wc2data_func = wordcloud.WordCloud(font_path = 'C:\\Windows\\Fonts\\msyh.ttc',
													relative_scaling = 0.2,
													prefer_horizontal = 0.9,
													width = 5600,
													height = 2800,
													mode = 'RGB',
													background_color = 'white',mask = imread(wc2png_path)).fit_words(wc2data_dic)

				
				# 指定单词组颜色(官网自定义函数)
				class GroupedColorFunc(object):
					def __init__(self,color_to_words,default_color):
						self.color_func_to_words = [
							(get_single_color_func(color),set(words))
							for (color,words) in color_to_words.items()]
						self.defalt_color_func = get_single_color_func(default_color)
					def get_color_func(self,word):
						try:
							color_func = next(color_func for (color_func,words) in self.color_func_to_words
											if word in words)
						except StopIteration:
							color_func = self.defalt_color_func
						return color_func
					def __call__(self,word,**kwargs):
						return self.get_color_func(word)(word,**kwargs)

				# 指定单词组颜色(literal_eval为将string格式转为dict格式)
				color_to_words = ast.literal_eval(text_color_to_words.get('0.0','end'))
				default_color = text_default_color.get('0.0','end').replace('\n','')
				grouped_color_func = GroupedColorFunc(color_to_words,default_color)
				wc2data_func.recolor(color_func = grouped_color_func)


				# 导出图片
				wc2data_func.to_file(wc2png_export_path)

				Label(WC2_frame,text = '导出成功！').place(x = 569,y = 540)
			except Exception as e:
				showinfo(title = 'word_cloud_2',message = '报错信息如下：\n \n' + traceback.format_exc())
			

		FR.destroy()

		WC2_frame = Frame(top,width = 1200,height = 600)
		WC2_frame.pack()
		WC2_frame.pack_propagate(False)
		wc2path = StringVar()
		WC2_brief = '                                                      词云生成（指定单词组颜色）\n'\
					'词云可以对网络文本中出现频率较高的“关键词”予以视觉上的突出，形成“关键词云层”或“关键词渲染”，从而过滤\n'\
					'掉大量的文本信息，使读者能够快速领略文本主旨。而“词云”这一概念最早是2006年由美国西北大学新闻学副教授、\n'\
					'新媒体专业主任里奇·戈登（Rich Gordon）提出。\n \n▪ 词频：即目标文本词汇出现频次（.xlsx）格式参考见'\
					'测试文件。\n▪ 轮廓图片：即生成词云形状的蒙版图片，请尽量选择边界内外颜色对比度强的图片（.png）格式参考见'\
					'测试文件。\n▪ 指定单词组颜色：给予需要突出显示的单词独特的颜色（可使用16进制色彩表）格式见下方文本框。\n'\
					'▪ 其他单词默认颜色：给予除突出显示外的其他单词独特的颜色（可使用16进制色彩表）格式见下方文本框。'
		Label(WC2_frame,text = WC2_brief).place(x = 18,y = 13)
		Label(WC2_frame,text = '导入词频路径：').place(x = 18,y = 192)
		Label(WC2_frame,text = '导入轮廓图片路径：').place(x = 18,y = 230)
		Label(WC2_frame,text = '导出图片路径：').place(x = 18,y = 272)
		wc2data_path_entry = Entry(WC2_frame)
		wc2data_path_entry.place(x = 168,y = 192)
		wc2png_path_entry = Entry(WC2_frame)
		wc2png_path_entry.place(x = 168,y = 230)
		wc2png_export_path_entry = Entry(WC2_frame,textvariable = wc2path)
		wc2png_export_path_entry.place(x = 168,y = 272)
		Button(WC2_frame,text = '导入词频',command = WC2openfile).place(x = 414,y = 192)
		Button(WC2_frame,text = '导入轮廓图片',command = WC2openfile1).place(x = 414,y = 230)
		Button(WC2_frame,text = '导出图片',command = WC2exportfile).place(x = 414,y = 272)
		Button(WC2_frame,text = '运行',command = word_cloud_2).place(x = 414,y = 540)

		Label(WC2_frame,text = '指定单词颜色格式示范：').place(x = 18,y = 314)
		Label(WC2_frame,text = '其他单词默认颜色：').place(x = 18,y = 540)
		text_color_to_words = Text(WC2_frame,height = 12,width = 50)
		text_color_to_words.place(x = 18,y = 353)
		text_color_to_words.insert(INSERT,"{'green':['便秘','恶心'],'red':['贫血','便血'],'yellow':['腹痛','腹泄']}")
		text_default_color = Text(WC2_frame,height = 1,width = 27)
		text_default_color.place(x = 180,y = 540)
		text_default_color.insert(INSERT,"gray")
		
		FR = WC2_frame

	def machine_learning_page():
		# 设置peak界面
	
		peak = Toplevel(top)
		peak.title('TAS-Test v-%s-Machine Learning'%version)
		peak.geometry('450x400+600+200')
		Label(peak,text = '尚未完善').pack()
		#peak.iconbitmap('.\\TAS.ico')

	def lantent_dirichlet_allocation_page():
		# 设置summit界面
		summit = Toplevel(top)
		summit.title('TAS-Test v-%s-Lantent Dirichlet Allocation'%version)
		summit.geometry('450x400+400+200')
		Label(summit,text = '尚未完善').pack()
		#summit.iconbitmap('.\\TAS.ico')


    # 定义菜单栏Naive Bayes按钮函数         
	def Naive_Bayes():

		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		# 定义Naive Bayes分类算法函数
		def model_naive_bayes():
			try:
				start = time.clock()

				NBdata_path = NBdata_path_entry.get()
				NBstopwords_path = NBstopwords_path_entry.get()

				data0 = pd.read_excel(NBdata_path)
				stop_words = list(pd.read_csv(NBstopwords_path,names = ['word'],
					sep = 'aaa',encoding = 'UTF-8-sig',engine='python').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) >1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				NBmodel = naive_bayes.MultinomialNB()
				NBmodel.fit(x_train,y_train)

				n_train = NBmodel.score(x_train,y_train)
				n_test = NBmodel.score(x_test,y_test)

				NBmodel_report = classification_report(y_test,NBmodel.predict(x_test))

				end = time.clock()

				Label(NB_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(NB_frame,text = NBmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(NB_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)
				

			except Exception as e:
				showinfo(title = 'naive_bayes',message = '报错信息如下：\n \n' + traceback.format_exc())

		# 定义导入文件函数
		def NBopenfile1():
			NBfilename1 = filedialog.askopenfilename(title = '导入数据')
			NBdata_path_entry.insert('insert',NBfilename1)

		def NBopenfile2():
			NBfilename2 = filedialog.askopenfilename(title = '导入词表')
			NBstopwords_path_entry.insert('insert',NBfilename2)

		FR.destroy() 

		NB_frame = Frame(top,width = 1200,height = 600)
		NB_frame.pack()
		NB_frame.pack_propagate(False)
		Label(NB_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(NB_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		NBdata_path_entry = Entry(NB_frame)
		NBdata_path_entry.grid(row = 0,column = 1,sticky = W)
		NBstopwords_path_entry = Entry(NB_frame)
		NBstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(NB_frame,text = '导入数据',command = NBopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(NB_frame,text = '导入词表',command = NBopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(NB_frame,text = '分析数据',command = model_naive_bayes).grid(row = 2,column = 2,sticky = E)

		FR = NB_frame


	# 定义菜单栏Logistic Regression按钮函数
	def Logistic_Regression():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_logistic_regression():
			try:
				start = time.clock()

				LRdata_path = LRdata_path_entry.get()
				LRstopwords_path = LRstopwords_path_entry.get()

				data0 = pd.read_excel(LRdata_path)
				stop_words = list(pd.read_csv(LRstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig',engine='python').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) >1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				LRmodel = LogisticRegression()
				LRmodel.fit(x_train,y_train)

				n_train = LRmodel.score(x_train,y_train)
				n_test = LRmodel.score(x_test,y_test)

				LRmodel_report = classification_report(y_test,LRmodel.predict(x_test))

				end = time.clock()

				Label(LR_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(LR_frame,text = LRmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(LR_frame,text = '训练的时间：%.2f Seconds' %(end - start)).grid(row = 5,column = 1,sticky = W)
				

			except Exception as e:
				showinfo(title = 'logistic_regression',message = '报错信息如下：\n \n' + traceback.format_exc())

			

		# 定义导入文件函数
		def LRopenfile1():
			LRfilename1 = filedialog.askopenfilename(title = '导入数据')
			LRdata_path_entry.insert('insert',LRfilename1)

		def LRopenfile2():
			LRfilename2 = filedialog.askopenfilename(title = '导入词表')
			LRstopwords_path_entry.insert('insert',LRfilename2)

		FR.destroy()

		LR_frame = Frame(top,width = 1200,height = 600)
		LR_frame.pack()
		LR_frame.pack_propagate(False)
		Label(LR_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(LR_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		LRdata_path_entry = Entry(LR_frame)
		LRdata_path_entry.grid(row = 0,column = 1,sticky = W)
		LRstopwords_path_entry = Entry(LR_frame)
		LRstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(LR_frame,text = '导入数据',command = LRopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(LR_frame,text = '导入词表',command = LRopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(LR_frame,text = '分析数据',command = model_logistic_regression).grid(row = 2,column = 2,sticky = E)
		
		FR = LR_frame


	# 定义菜单栏Support Vector Machine按钮函数
	def Support_Vector_Machine():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_support_vector_machine():
			try:
				start = time.clock()

				SVMdata_path = SVMdata_path_entry.get()
				SVMstopwords_path = SVMstopwords_path_entry.get()

				data0 = pd.read_excel(SVMdata_path)
				stop_words = list(pd.read_csv(SVMstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				SVMmodel = SVC(C = 10.0,gamma = 0.01,kernel = 'linear')
				SVMmodel.fit(x_train,y_train)

				n_train = SVMmodel.score(x_train,y_train)
				n_test = SVMmodel.score(x_test,y_test)

				SVMmodel_report = classification_report(y_test,SVMmodel.predict(x_test))

				end = time.clock()

				Label(SVM_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(SVM_frame,text = SVMmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(SVM_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)


			except Exception as e:
				showinfo(title = 'support_vector_machine',message = '报错信息如下：\n \n' + traceback.format_exc())


		# 定义导入文件函数
		def SVMopenfile1():
			SVMfilename1 = filedialog.askopenfilename(title = '导入数据')
			SVMdata_path_entry.insert('insert',SVMfilename1)

		def SVMopenfile2():
			SVMfilename2 = filedialog.askopenfilename(title = '导入词表')
			SVMstopwords_path_entry.insert('insert',SVMfilename2)

		FR.destroy()

		SVM_frame = Frame(top,width = 1200,height = 600)
		SVM_frame.pack()
		SVM_frame.pack_propagate(False)
		Label(SVM_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(SVM_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		SVMdata_path_entry = Entry(SVM_frame)
		SVMdata_path_entry.grid(row = 0,column = 1,sticky = W)
		SVMstopwords_path_entry = Entry(SVM_frame)
		SVMstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(SVM_frame,text = '导入数据',command = SVMopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(SVM_frame,text = '导入词表',command = SVMopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(SVM_frame,text = '分析数据',command = model_support_vector_machine).grid(row = 2,column = 2,sticky = E)

		FR = SVM_frame



	# 定义菜单栏Decision Tree按钮函数   
	def Decision_Tree():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_decision_tree():
			try:
				start = time.clock()

				DTdata_path = DTdata_path_entry.get()
				DTstopwords_path = DTstopwords_path_entry.get()

				data0 = pd.read_excel(DTdata_path)
				stop_words = list(pd.read_csv(DTstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				DTmodel = tree.DecisionTreeClassifier(criterion = 'entropy')
				DTmodel.fit(x_train,y_train)

				n_train = DTmodel.score(x_train,y_train)
				n_test = DTmodel.score(x_test,y_test)

				DTmodel_report = classification_report(y_test,DTmodel.predict(x_test))

				end = time.clock()

				Label(DT_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(DT_frame,text = DTmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(DT_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)

			except Exception as e:
				showinfo(title = 'decision_tree',message = '报错信息如下：\n \n' + traceback.format_exc())

		# 定义导入文件函数
		def DTopenfile1():
			DTfilename1 = filedialog.askopenfilename(title = '导入数据')
			DTdata_path_entry.insert('insert',DTfilename1)

		def DTopenfile2():
			DTfilename2 = filedialog.askopenfilename(title = '导入词表')
			DTstopwords_path_entry.insert('insert',DTfilename2)


		FR.destroy()
        
		DT_frame = Frame(top,width = 1200,height = 600)
		DT_frame.pack()
		DT_frame.pack_propagate(False)
		Label(DT_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(DT_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		DTdata_path_entry = Entry(DT_frame)
		DTdata_path_entry.grid(row = 0,column = 1,sticky = W)
		DTstopwords_path_entry = Entry(DT_frame)
		DTstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(DT_frame,text = '导入数据',command = DTopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(DT_frame,text = '导入词表',command = DTopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(DT_frame,text = '分析数据',command = model_decision_tree).grid(row = 2,column = 2,sticky = E)
		
		FR = DT_frame        



	def Adaptive_Boosting():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_adaptive_boosting():
			try:
				start = time.clock()

				ABdata_path = ABdata_path_entry.get()
				ABstopwords_path = ABstopwords_path_entry.get()

				data0 = pd.read_excel(ABdata_path)
				stop_words = list(pd.read_csv(ABstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				ABmodel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, 
					min_samples_leaf=5),algorithm = 'SAMME',n_estimators = 200,learning_rate = 0.8)
				ABmodel.fit(x_train,y_train)

				n_train = ABmodel.score(x_train,y_train)
				n_test = ABmodel.score(x_test,y_test)

				ABmodel_report = classification_report(y_test,ABmodel.predict(x_test))

				end = time.clock()

				Label(AB_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(AB_frame,text = ABmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(AB_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)

			except Exception as e:
				showinfo(title = 'adaptive_boosting',message = '报错信息如下：\n \n' + traceback.format_exc())

		# 定义导入文件函数
		def ABopenfile1():
			ABfilename1 = filedialog.askopenfilename(title = '导入数据')
			ABdata_path_entry.insert('insert',ABfilename1)

		def ABopenfile2():
			ABfilename2 = filedialog.askopenfilename(title = '导入词表')
			ABstopwords_path_entry.insert('insert',ABfilename2)

		FR.destroy()

		AB_frame = Frame(top,width = 1200,height = 600)
		AB_frame.pack()
		AB_frame.pack_propagate(False)
		Label(AB_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(AB_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		ABdata_path_entry = Entry(AB_frame)
		ABdata_path_entry.grid(row = 0,column = 1,sticky = W)
		ABstopwords_path_entry = Entry(AB_frame)
		ABstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(AB_frame,text = '导入数据',command = ABopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(AB_frame,text = '导入词表',command = ABopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(AB_frame,text = '分析数据',command = model_adaptive_boosting).grid(row = 2,column = 2,sticky = E)

		FR = AB_frame        




	def Gradient_Boosting_Decision_Tree():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_gradient_boosting_decision_tree():
			try:
				start = time.clock()

				GBDTdata_path = GBDTdata_path_entry.get()
				GBDTstopwords_path = GBDTstopwords_path_entry.get()

				data0 = pd.read_excel(GBDTdata_path)
				stop_words = list(pd.read_csv(GBDTstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				GBDTmodel = GradientBoostingClassifier(random_state = 111)
				GBDTmodel.fit(x_train,y_train)

				n_train = GBDTmodel.score(x_train,y_train)
				n_test = GBDTmodel.score(x_test,y_test)

				GBDTmodel_report = classification_report(y_test,GBDTmodel.predict(x_test))

				end = time.clock()

				Label(GBDT_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(GBDT_frame,text = GBDTmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(GBDT_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)

			except Exception as e:
				showinfo(title = 'gradient_boosting_decision_tree',message = '报错信息如下：\n \n' + traceback.format_exc())

		# 定义导入文件函数
		def GBDTopenfile1():
			GBDTfilename1 = filedialog.askopenfilename(title = '导入数据')
			GBDTdata_path_entry.insert('insert',GBDTfilename1)

		def GBDTopenfile2():
			GBDTfilename2 = filedialog.askopenfilename(title = '导入词表')
			GBDTstopwords_path_entry.insert('insert',GBDTfilename2)

		FR.destroy()

		GBDT_frame = Frame(top,width = 1200,height = 600)
		GBDT_frame.pack()
		GBDT_frame.pack_propagate(False)
		Label(GBDT_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(GBDT_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		GBDTdata_path_entry = Entry(GBDT_frame)
		GBDTdata_path_entry.grid(row = 0,column = 1,sticky = W)
		GBDTstopwords_path_entry = Entry(GBDT_frame)
		GBDTstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(GBDT_frame,text = '导入数据',command = GBDTopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(GBDT_frame,text = '导入词表',command = GBDTopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(GBDT_frame,text = '分析数据',command = model_gradient_boosting_decision_tree).grid(row = 2,column = 2,sticky = E)
		
		FR = GBDT_frame




	def eXtreme_Gradient_Boosting():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_extreme_gradient_boosting():
			try:
				start = time.clock()

				XGBdata_path = XGBdata_path_entry.get()
				XGBstopwords_path = XGBstopwords_path_entry.get()

				data0 = pd.read_excel(XGBdata_path)
				stop_words = list(pd.read_csv(XGBstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				XGBmodel = GradientBoostingClassifier(random_state = 111)
				XGBmodel.fit(x_train,y_train)

				n_train = XGBmodel.score(x_train,y_train)
				n_test = XGBmodel.score(x_test,y_test)

				XGBmodel_report = classification_report(y_test,XGBmodel.predict(x_test))

				end = time.clock()

				Label(XGB_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(XGB_frame,text = XGBmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(XGB_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)

			except Exception as e:
				showinfo(title = 'extreme_gradient_boosting',message = '报错信息如下：\n \n' + traceback.format_exc())

		# 定义导入文件函数
		def XGBopenfile1():
			XGBfilename1 = filedialog.askopenfilename(title = '导入数据')
			XGBdata_path_entry.insert('insert',XGBfilename1)

		def XGBopenfile2():
			XGBfilename2 = filedialog.askopenfilename(title = '导入词表')
			XGBstopwords_path_entry.insert('insert',XGBfilename2)


		FR.destroy()

		XGB_frame = Frame(top,width = 1200,height = 600)
		XGB_frame.pack()
		XGB_frame.pack_propagate(False)
		Label(XGB_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(XGB_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		XGBdata_path_entry = Entry(XGB_frame)
		XGBdata_path_entry.grid(row = 0,column = 1,sticky = W)
		XGBstopwords_path_entry = Entry(XGB_frame)
		XGBstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(XGB_frame,text = '导入数据',command = XGBopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(XGB_frame,text = '导入词表',command = XGBopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(XGB_frame,text = '分析数据',command = model_extreme_gradient_boosting).grid(row = 2,column = 2,sticky = E)

		FR = XGB_frame





	def Random_Forest():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		def model_random_forest():
			try:
				start = time.clock()

				RFdata_path = RFdata_path_entry.get()
				RFstopwords_path = RFstopwords_path_entry.get()

				data0 = pd.read_excel(RFdata_path)
				stop_words = list(pd.read_csv(RFstopwords_path,
					names = ['word'],sep = 'aaa',encoding = 'UTF-8-sig').word)

				def m_cut(intxt):
					return [w for w in jieba.lcut(intxt) if w not in stop_words and len(w) > 1]

				data0_part = [' '.join(m_cut(w)) for w in data0.comment]

				countvec = CountVectorizer(ngram_range = (1,2))
				x = countvec.fit_transform(data0_part)

				x_train,x_test,y_train,y_test = train_test_split(x,data0.feel,
					test_size = 0.2,random_state = 111)

				RFmodel = GradientBoostingClassifier(random_state = 111)
				RFmodel.fit(x_train,y_train)

				n_train = RFmodel.score(x_train,y_train)
				n_test = RFmodel.score(x_test,y_test)

				RFmodel_report = classification_report(y_test,RFmodel.predict(x_test))

				end = time.clock()

				Label(RF_frame,text = '训练集得分：%f \n验证集集得分：%f' %(n_train,n_test)).grid(row = 3,column = 1,sticky = W)
				Label(RF_frame,text = RFmodel_report).grid(row = 4,column = 1,sticky = W)
				Label(RF_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 5,column = 1,sticky = W)

			except Exception as e:
				showinfo(title = 'random_forest',message = '报错信息如下：\n \n' + traceback.format_exc())

		# 定义导入文件函数
		def RFopenfile1():
			RFfilename1 = filedialog.askopenfilename(title = '导入数据')
			RFdata_path_entry.insert('insert',RFfilename1)

		def RFopenfile2():
			RFfilename2 = filedialog.askopenfilename(title = '导入词表')
			RFstopwords_path_entry.insert('insert',RFfilename2)

		FR.destroy()

		RF_frame = Frame(top,width = 1200,height = 600)
		RF_frame.pack()
		RF_frame.pack_propagate(False)
		Label(RF_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(RF_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		RFdata_path_entry = Entry(RF_frame)
		RFdata_path_entry.grid(row = 0,column = 1,sticky = W)
		RFstopwords_path_entry = Entry(RF_frame)
		RFstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		Button(RF_frame,text = '导入数据',command = RFopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(RF_frame,text = '导入词表',command = RFopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(RF_frame,text = '分析数据',command = model_random_forest).grid(row = 2,column = 2,sticky = E)
		
		FR = RF_frame


	def Lantent_Dirichlet_Allocation():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame


		def model_lda():
			try:
				start = time.clock()

				# 导入文本
				LDAdata_path = LDAdata_path_entry.get()
				LDAstopwords_path = LDAstopwords_path_entry.get()
				LDAloaduserdict_path = LDAloaduserdict_path_entry.get()
				txt_0 = pd.read_table(LDAdata_path,names = ['text'],encoding = 'UTF-8-sig',engine='python')
				stop_words = list(pd.read_csv(LDAstopwords_path,names = ['word'],sep = 'aaa',encoding = 'UTF-8',engine='python').word)
				jieba.load_userdict(LDAloaduserdict_path)

				# 去除文本分隔符
				lst = list(txt_0['text'])
				lst1 = []
				for i in range(len(lst)):
					s = lst[i].replace('|||','')
					lst1.append(s)
				txt = pd.DataFrame(lst1,columns = ['text'])

				# 自定义分词函数
				def m_cut(intxt):
					return [ w for w in jieba.cut(intxt) if w not in stop_words and len(w) > 1]

				# 生成term-doc矩阵
				txt_part = [' '.join(m_cut(w)) for w in txt.text]
				countvec = CountVectorizer(min_df = 0.05)
				x = countvec.fit_transform(txt_part)

				# LDA模型拟合
				LDAmodel = LatentDirichletAllocation(n_topics = 7,max_iter = 50,random_state = 111)
				LDAmodel.fit(x)

				end = time.clock()

			#输出LDA模型
				# LDA模型困惑度
				n_perplexity = LDAmodel.perplexity(x)
				# 定义输出主题模型分布函数
				def LDAmodel_report(model,feature_names,n_top_words):
					LDAmodel_report_list = []
					for topic_idx,topic in enumerate(model.components_):
						topic_title = 'Topic #%d:\n' % topic_idx
						topic_words = ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
						topic_titleword = topic_title + topic_words
						LDAmodel_report_list.append(topic_titleword)
					return LDAmodel_report_list

				n_top_words = 12
				LDAmodel_feature_names = countvec.get_feature_names()
				LDAmodel_report = '\n'.join(LDAmodel_report(LDAmodel,LDAmodel_feature_names,n_top_words))

				Label(LDA_frame,text = '训练困惑度：%f' % n_perplexity).grid(row = 4,column = 1)
				Label(LDA_frame,text = LDAmodel_report).grid(row = 5,column = 1)
				Label(LDA_frame,text = '训练的时间：%.2f Seconds' % (end - start)).grid(row = 6,column = 1)
				

			except Exception as e:
				showinfo(title = 'lantent_dirichlet_allocation',message = '报错信息如下：\n \n' + traceback.format_exc())	         	



		def LDAopenfile1():
			LDAfilename1 = filedialog.askopenfilename(title = '导入数据')
			LDAdata_path_entry.insert('insert',LDAfilename1)

		def LDAopenfile2():
			LDAfilename2 = filedialog.askopenfilename(title = '导入词表')
			LDAstopwords_path_entry.insert('insert',LDAfilename2)

		def LDAopenfile3():
			LDAfilename3 = filedialog.askopenfilename(title = '导入词库')
			LDAloaduserdict_path_entry.insert('insert',LDAfilename3)



		FR.destroy()

		LDA_frame = Frame(top,width = 1200,height = 600)
		LDA_frame.pack()
		LDA_frame.pack_propagate(False)
		Label(LDA_frame,text = '数据路径：').grid(row = 0,column = 0,sticky = W)
		Label(LDA_frame,text = '词表路径：').grid(row = 1,column = 0,sticky = W)
		Label(LDA_frame,text = '词库路径：').grid(row = 2,column = 0,sticky = W)
		LDAdata_path_entry = Entry(LDA_frame)
		LDAdata_path_entry.grid(row = 0,column = 1,sticky = W)
		LDAstopwords_path_entry = Entry(LDA_frame)
		LDAstopwords_path_entry.grid(row = 1,column = 1,sticky = W)
		LDAloaduserdict_path_entry = Entry(LDA_frame)
		LDAloaduserdict_path_entry.grid(row = 2,column = 1,sticky = W)
		Button(LDA_frame,text = '导入数据',command = LDAopenfile1).grid(row = 0,column = 2,sticky = E)
		Button(LDA_frame,text = '导入词表',command = LDAopenfile2).grid(row = 1,column = 2,sticky = E)
		Button(LDA_frame,text = '导入词库',command = LDAopenfile3).grid(row = 2,column = 2,sticky = E)
		Button(LDA_frame,text = '分析数据',command = model_lda).grid(row = 3,column = 2,sticky = E)
		FR = LDA_frame


	def Update_Log():
		global FR,init_frame,LR_frame,SVM_frame,NB_frame,DT_frame,AB_frame, \
		GBDT_frame,XGB_frame,RF_frame,LDA_frame,WF_frame,WC1_frame,WC2_frame, \
		HM_frame,UL_frame

		FR.destroy()

		UL_frame = Frame(top,width = 1200,height = 600)
		UL_frame.pack()
		UL_frame.pack_propagate(False)

		# 更新日志内容
		UL_frame_text0 = '\n \n \n TAS-test-v0.4.1（Text Anatomy Software）\n 18 August 2019'\
		'\n ▪ 新增“更新日志”页面\n ▪ 新增“指定单词组颜色”生成词云功能（高级功能尚未完善）\n ▪ 更改了主页面尺寸'
		UL_frame_text1 = '\n \n \n TAS-test-v0.4.2（Text Anatomy Software）\n 19 August 2019'\
		'\n ▪ 修复了各个模块的BUG\n ▪ 测试了win10 1903系统的稳定性'
		UL_frame_text2 = '\n \n \n TAS-test-v0.4.3（Text Anatomy Software）\n 20 August 2019'\
		'\n ▪ 优化了代码结构\n ▪ 新增“Logistic Regression”功能\n ▪ 测试了win10 专业版系统的稳定性'
		UL_frame_text3 = '\n \n \n TAS-test-v0.4.4（Text Anatomy Software）\n 21 August 2019'\
		'\n ▪ 调整了菜单栏的功能顺序及代码顺序\n ▪ 新增“指定单词组颜色”生成词云高级功能\n ▪ 对“指定单词组颜色”'\
		'生成词云页面进行重新布置'
		UL_frame_text4 = '\n \n \n TAS-test-v0.4.5（Text Anatomy Software）\n 22 August 2019'\
		'\n ▪ 调整了菜单栏结构\n ▪ 新增“模型调优”菜单\n ▪ 更新了首页显示内容\n ▪ 新增“Support Vector '\
		'Machine”功能\n ▪ 新增“Decision Tree”功能\n ▪ 将“更新日志”下显示框更换为只读文本框'
		UL_frame_text5 = '\n \n \n TAS-test-v0.4.6（Text Anatomy Software）\n 23 August 2019'\
		'\n ▪ 调整了Traditional Learning下各个算法的输出格式\n ▪ 调整了LDA算法的输出格式\n ▪ 修改了各个页面标题'\
		'\n ▪ 新增了版本号的同步规则\n ▪ 新增了详细报错信息\n ▪ 调整了菜单栏结构'
		UL_frame_text6 = '\n \n \n TAS-test-v0.4.7（Text Anatomy Software）\n 24 August 2019'\
		'\n ▪ 新增“Gradient Boosting Decision Tree”功能\n ▪ 新增“eXtreme Gradient Boosting”功能\n'\
		' ▪ 新增“Random Forest”功能'
		UL_frame_text7 = ' TAS-test-v0.4.8（Text Anatomy Software）\n 26 August 2019'\
		'\n ▪ 新增“Adaptive Boosting”功能'
		
		UL_frame_text = UL_frame_text7 + UL_frame_text6 + UL_frame_text5 + \
						UL_frame_text4 + UL_frame_text3 + UL_frame_text2 + \
						UL_frame_text1 + UL_frame_text0
		text_UL_frame = Text(UL_frame,width=91, height=45)
		text_UL_frame.place(x = 0,y = 0)
		text_UL_frame.insert(INSERT,UL_frame_text)
		text_UL_frame.config(state = DISABLED)

		FR = UL_frame




	# 设置top界面    
	win.withdraw()
	top = Toplevel(win)
	top.title('TAS(Text Anatomy Software)-Test v-%s'%version)
	top.geometry('650x620+600+200')
	#top.iconbitmap('.\\TAS.ico')


	# 设置top界面菜单栏
	menuBar = Menu(top)


	menu1 = Menu(menuBar,tearoff = 0)
	menuBar.add_cascade(label = '数据处理',menu = menu1)
	menu1.add_command(label = 'Word Frequency',command = Word_Frequency)
	menu1.add_command(label = 'Hospital Match',command = Hospital_Match)
	menu_Word_Cloud = Menu(menu1,tearoff = 0)
	menu1.add_cascade(label = 'Word Cloud',menu = menu_Word_Cloud)
	menu_Word_Cloud.add_command(label = '色系模仿',command = Word_Cloud_1)
	menu_Word_Cloud.add_command(label = '指定单词组颜色',command = Word_Cloud_2)

	menu2 = Menu(menuBar,tearoff = 0)
	menuBar.add_cascade(label = '模型调优',menu = menu2)
	menu2.add_command(label = 'Machine Learning',command = machine_learning_page)
	menu2.add_command(label = 'Lantent Dirichlet Allocation',command = lantent_dirichlet_allocation_page)

	

	menu3 = Menu(menuBar,tearoff = 0)
	menuBar.add_cascade(label = '结果输出',menu = menu3)

	menu_traditional_learning = Menu(menu3,tearoff = 0)
	menu3.add_cascade(label = 'Traditional Learning',menu = menu_traditional_learning)
	menu_traditional_learning.add_command(label = 'Naive Bayes',command = Naive_Bayes)
	menu_traditional_learning.add_command(label = 'Logistic Regression',command = Logistic_Regression)
	menu_traditional_learning.add_command(label = 'Support Vector Machine',command = Support_Vector_Machine)
	menu_traditional_learning.add_command(label = 'Decision Tree',command = Decision_Tree)
	
	menu_ensemble_learning = Menu(menu3,tearoff = 0)
	menu3.add_cascade(label = 'Ensemble Learning',menu = menu_ensemble_learning)
	menu_ensemble_learning.add_command(label = 'Adaptive Boosting',command = Adaptive_Boosting)
	menu_ensemble_learning.add_command(label = 'Gradient Boosting Decision Tree',command = Gradient_Boosting_Decision_Tree)
	menu_ensemble_learning.add_command(label = 'eXtreme Gradient Boosting',command = eXtreme_Gradient_Boosting)
	menu_ensemble_learning.add_command(label = 'Random Forest',command = Random_Forest)

	menu_probabilistic_graphical_model = Menu(menu3,tearoff = 0)
	menu3.add_cascade(label = 'Probabilistic Graphical Model',menu = menu_probabilistic_graphical_model)
	menu_probabilistic_graphical_model.add_command(label = 'Lantent Dirichlet Allocation',command = Lantent_Dirichlet_Allocation)


	menu4 = Menu(menuBar,tearoff = 0)
	menuBar.add_cascade(label = '更多',menu = menu4)
	menu4.add_command(label = '设置',command = myMessage0)
	menu4.add_command(label = '帮助',command = myMessage1)
	menu4.add_command(label = '更新日志',command = Update_Log)
	menu4.add_command(label = '关于',command = myMessage2)

	menuBar.add_command(label = '退出',command = win.destroy)
	
	top['menu'] = menuBar
	

	# 新建初始化frame，供后续跳转
	init_frame = Frame(top,width = 1200,height = 600)
	init_frame.pack()
	init_frame.pack_propagate(False)

	init_title = '       欢迎使用TAS（Text Anatomy Software）'
	Label(init_frame,text = init_title,anchor = 'center',
		font = ('微软雅黑',20)).place(x = 0,y = 100)

	init_warning = '      警告：关闭页面请勿点击右上角，请点击菜单栏中的“退出”'
	Label(init_frame,text = init_warning,anchor = 'center',
		font = ('微软雅黑',16),foreground = 'red').place(x = 0,y = 220)

	init_brief = '    TAS全称：Text Anatomy Software。本工具主要功能为文本挖掘，包括有监督学习及\n'\
	'    无监督学习。此外包括数据处理的部分功能。'
	Label(init_frame,text = init_brief,font = ('微软雅黑',12)).place(x = 0,y = 380)

	FR = init_frame






win = Tk()
win.title('TAS(Text Anatomy Software)-Test v-%s'%version)
win.geometry('600x300+800+350')
#win.iconbitmap('.\\TAS.ico')




canvas = Canvas(win,height = 136,width = 500)
image_file = PhotoImage(file = 'welcome.png')
image = canvas.create_image(0,0,anchor = 'nw',image = image_file)
canvas.pack(side = 'top')


Label(win,text = 'Username:').place(x = 170,y = 150)
Label(win,text = 'Password:').place(x = 170,y = 190)


entry_username = Entry(win)
entry_username.place(x = 280,y = 150)
entry_password = Entry(win,show = '*')
entry_password.place(x = 280,y = 190)

Button(win,text = 'Login',command = userlogin).place(x = 170,y = 230)
Button(win,text = 'Exit',command = win.destroy).place(x = 340,y = 230)



win.mainloop()