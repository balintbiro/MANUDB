#importations
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
taxonomy=pd.read_csv('../taxonomy.txt',index_col=0)

grouped=taxonomy.groupby('Order ')

#generate the input data for the sunburst diagram
final_container=[]
summation=0
for order in grouped.groups:
    subdf=grouped.get_group(order)
    fams=subdf['Family ']
    famsU=fams.unique()
    order_container=[]
    for fam in famsU:
        subsubdf=subdf[subdf['Family ']==fam]
        gens=subsubdf['Genus '].unique()
        if len(gens)>1:
            fam_container=[]
            for gen in gens:
                genus=gen.strip()
                genusSize=int(subsubdf['Genus '].value_counts().get(gen))
                fam_container.append((f"{genus} ({genusSize})",genusSize,[]))
            family=fam.strip()
            familySize=int(subdf['Family '].value_counts().get(fam))
            order_container.append((f"{family} ({familySize})",familySize,fam_container))
    if len(order_container)>1:
        orderSize=subdf.shape[0]
        final_container.append((f"{order} ({orderSize})",orderSize,order_container))
        summation+=subdf.shape[0]

data=[('MANUDB',summation,final_container)]

#function for plotting sunburst figure
def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    levelColors={0:"white",1:"lightblue",2:"lightgreen",3:"bisque"}
    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2],color=levelColors[level])
        ax.text(0, 0, label, ha='center', va='center',fontsize=20)
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='black', align='edge',color=levelColors[level])
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center') 
    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()

fig,ax=plt.subplots(1,1,figsize=(14,14),subplot_kw={'projection': 'polar'})
sunburst(nodes=data,ax=ax)
plt.savefig("Fig1.png",dpi=400)