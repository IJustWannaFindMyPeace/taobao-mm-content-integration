"""Render portable PNG/SVG figures from the published, audited evidence bundle."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--font',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];muse=root.parent/'taobao-mm-muse-repro'
    evidence=root/'reports/storyline-evidence.json';data=json.loads(evidence.read_text(encoding='utf-8'))
    font=a.font or Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));fontname=font_manager.FontProperties(fname=str(font)).get_name()
    else:fontname='Noto Sans CJK SC'
    plt.rcParams.update({'font.family':fontname,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'white'})
    colors=['#2563a6','#d07828','#20806b'];seeds=data['seeds'];manifest=[]
    output=root/'figures/storyline';output.mkdir(parents=True,exist_ok=True)
    for key,spec in data['figures'].items():
        rows=spec['rows'];kind=spec['kind'];labels=list(dict.fromkeys(r['label'] for r in rows))
        if kind=='metrics':
            fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True)
            for ax,metric in zip(axes,['gauc','auc','logloss']):
                for i,label in enumerate(labels):
                    values=[r[metric] for r in rows if r['label']==label]
                    ax.errorbar(np.mean(values),i,xerr=np.std(values,ddof=1),fmt='o',color=colors[0],capsize=4)
                ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
                ax.set_title(metric.upper()+(' ↓' if metric=='logloss' else ' ↑'));ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                ax.xaxis.set_major_locator(MaxNLocator(4))
            # A shared y-axis needs only one inversion.
            axes[0].set_ylim(len(labels)-.5,-.5)
            fig.subplots_adjust(left=.29,right=.98,wspace=.2,top=.85,bottom=.20)
        elif kind=='training':
            fig,axes=plt.subplots(2,2,figsize=(13,8))
            for j,label in enumerate(labels):
                for seed,color in zip(seeds,colors):
                    group=sorted([r for r in rows if r['label']==label and r['seed']==seed],key=lambda r:r['epoch'])
                    x=[r['epoch'] for r in group]
                    axes[0,j].plot(x,[r['gauc'] for r in group],'-o',color=color,label=str(seed))
                    axes[1,j].plot(x,[r['logloss'] for r in group],'-o',color=color)
                    axes[1,j].plot(x,[r['online_train_logloss'] for r in group],'--',color=color)
                axes[0,j].set_title(label+('（batch=256）' if j==0 else '（batch=32）'));axes[0,j].set_ylabel('验证 GAUC ↑')
                axes[1,j].set_ylabel('LogLoss ↓');axes[1,j].set_xlabel('训练轮数；实线验证／虚线在线训练')
                for ax in axes[:,j]:ax.set_xticks([1,2,3]);ax.grid(alpha=.2)
                axes[0,j].legend(title='训练种子',fontsize=9)
            fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.18,hspace=.40,wspace=.24)
        elif kind in ('epochs','length'):
            fig,ax=plt.subplots(figsize=(12,6.5));axes=[ax]
            if kind=='epochs':
                for seed,color in zip(seeds,colors):
                    group=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['epoch'])
                    ax.scatter([0],[group[0]['gauc']],marker='s',s=60,color=color)
                    ax.plot([r['epoch'] for r in group[1:]],[r['gauc'] for r in group[1:]],'-o',color=color,label=str(seed))
                ax.set_xticks([0,1,3,5],['P0：随机联合','P1：预训练冻结','P3：预训练冻结','P5：预训练冻结']);ax.legend(title='训练种子')
            else:
                for label,color in zip(labels,[colors[1],colors[0]]):
                    group=[r for r in rows if r['label']==label]
                    for seed in seeds:
                        rr=sorted([r for r in group if r['seed']==seed],key=lambda r:r['length'])
                        ax.plot([r['length'] for r in rr],[r['gauc'] for r in rr],color=color,alpha=.30,lw=1)
                    lengths=[50,200,500,1000]
                    ax.plot(lengths,[np.mean([r['gauc'] for r in group if r['length']==x]) for x in lengths],'-o',color=color,lw=2.8,label=label)
                ax.set_xticks([50,200,500,1000]);ax.set_xlabel('GSU候选历史上限 L（ESU处理数量 K=50）');ax.legend()
            ax.set_ylabel('测试 GAUC ↑');ax.grid(alpha=.2);fig.subplots_adjust(left=.10,right=.97,top=.85,bottom=.23)
        elif kind=='forest':
            fig,ax=plt.subplots(figsize=(12,6.5));ys=[]
            for i,r in enumerate(rows):
                color=colors[seeds.index(r['seed'])];x=r['delta_gauc']
                ax.errorbar(x,i,xerr=[[x-r['ci_low']],[r['ci_high']-x]],fmt='o',capsize=5,color=color)
                ys.append(r['label']+' · '+str(r['seed']))
            ax.set_yticks(range(len(rows)),ys);ax.invert_yaxis();ax.axvline(0,color='#4b5563',ls='--');ax.grid(axis='x',alpha=.2)
            ax.set_xlabel('测试 ΔGAUC（候选 − 内容余弦）');fig.subplots_adjust(left=.25,right=.97,top=.85,bottom=.23)
        elif kind=='frequency':
            fig,ax=plt.subplots(figsize=(12,6.5));freqs=list(dict.fromkeys(r['frequency'] for r in rows))
            for label,color in zip(labels,colors):
                group=[r for r in rows if r['label']==label]
                ax.plot(range(6),[np.mean([r['delta_auc'] for r in group if r['frequency']==f]) for f in freqs],'-o',label=label,color=color,lw=2)
                for seed in seeds:
                    ax.scatter(range(6),[next(r['delta_auc'] for r in group if r['frequency']==f and r['seed']==seed) for f in freqs],s=14,alpha=.4,color=color)
            ax.set_xticks(range(6),freqs);ax.set_xlabel('训练目标商品曝光频次');ax.set_ylabel('测试 ΔAUC（绝对差）');ax.legend();ax.grid(alpha=.2)
            fig.subplots_adjust(left=.10,right=.97,top=.85,bottom=.23)
        elif kind=='behavior':
            fig,axes=plt.subplots(1,2,figsize=(12,6.5));rows=sorted(rows,key=lambda r:seeds.index(r['seed']))
            for ax,metric,title in zip(axes,['negative_slope_fraction','overlap_fraction'],['内容相似度负斜率比例','与内容余弦Top-50的重叠率']):
                ax.bar([str(r['seed']) for r in rows],[100*r[metric] for r in rows],color=colors,width=.55)
                ax.set_ylim(0,110);ax.set_ylabel('%');ax.set_title(title);ax.set_xlabel('训练种子');ax.grid(axis='y',alpha=.2)
            fig.subplots_adjust(left=.08,right=.98,wspace=.25,top=.83,bottom=.24)
        elif kind=='diagram':
            fig,ax=plt.subplots(figsize=(14,7));ax.set_xlim(0,14);ax.set_ylim(0,7);ax.axis('off')
            def box(x,y,w,h,text,color):
                ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13',facecolor=color,edgecolor='none'))
                ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=13,color='#14243b')
            def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':2,'color':'#53677f'})
            box(.3,2.4,2.5,1.8,'共同输入\n目标广告＋用户历史\n固定内容向量／ID','#e4edf8')
            box(3.6,4.1,4.0,1.7,'项目一：内容如何接入？\n向量／相似度／统计\nMAKE预训练与接入','#e5f3ed')
            box(3.6,1.0,2.6,1.8,'项目二：GSU筛选\n从 L 条历史\n选择 Top-K','#fff0db')
            box(7.0,1.0,3.1,1.8,'ESU兴趣聚合\n统计／目标注意力\n语义感知注意力','#e5f3ed')
            box(11.0,2.4,2.4,1.8,'点击预测\nAUC／GAUC\nLogLoss','#e4edf8')
            arrow(2.8,3.9,3.6,4.8);arrow(2.8,2.7,3.6,1.9);arrow(6.2,1.9,7.0,1.9)
            arrow(7.6,4.8,11,3.9);arrow(10.1,1.9,11,2.7)
            ax.text(7,6.45,'衔接的是研究问题与共同协议，不是把两项目机械串成一个模型',ha='center',fontsize=13)
            fig.subplots_adjust(left=.02,right=.98,top=.85,bottom=.16)
        else:raise ValueError(kind)
        fig.suptitle(spec['title'],fontsize=19,fontweight='bold',x=.5,y=.96)
        note=spec['note']
        if kind=='metrics' and '标准差' not in note:
            note+=' 点为三种子均值，短线为样本标准差，非置信区间；横轴为局部范围。'
        wrapped='\n'.join(note[i:i+68] for i in range(0,len(note),68))
        fig.text(.04,.045,wrapped,fontsize=10,color='#475569',va='bottom')
        files={}
        for extension in ('png','svg'):
            path=output/(key+'.'+extension);fig.savefig(path,dpi=220)
            if extension=='svg':path.write_bytes(b'\n'.join(line.rstrip(b' \t') for line in path.read_bytes().splitlines())+b'\n')
            files[extension]=dict(path='figures/storyline/'+path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        plt.close(fig);manifest.append(dict(id=key,title=spec['title'],data_rows=len(spec['rows']),files=files))
    manifest=dict(scope='storyline_figure_assets',data_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=manifest,
                  note='Plots use local axis ranges; marker intervals are sample SD unless explicitly labeled bootstrap CI.')
    (root/'reports/storyline-figure-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (muse/'figures/storyline').mkdir(parents=True,exist_ok=True)
    for file in output.iterdir():shutil.copy2(file,muse/'figures/storyline'/file.name)
    shutil.copy2(root/'reports/storyline-figure-manifest.json',muse/'reports/storyline-figure-manifest.json')
    print('Rendered and copied 12 PNG + 12 SVG figures.')


if __name__=='__main__':main()
