import numpy as np
import matplotlib.pyplot as plt

data = [
    ('MANUDB', 115, [
        ('Carnivora', 32, [
            ('Felidae', 12, [
                ('Panthera',3,[]),
                ('Lynx',2,[]),
                ('Prionailurus',2,[]),
                ('Puma',2,[]),
                ('Acinonysx',1,[]),
                ('Felis',1,[]),
                ('Leopardus',1,[])
            ]),
            ('Ursidae', 4, [
                ('Ursus',3,[]),
                ('Ailuropoda',1,[]),
            ]),
            ('Phocidae', 4, [
                ('Halichoerus',1,[]),
                ('Leptonychotes',1,[]),
                ('Mirounga',1,[]),
                ('Phoca',1,[])
            ]),
            ('Mustelidae', 4, [
                ('Mustela',2,[]),
                ('Meles',1,[]),
                ('Lutra',1,[])
            ]),
            ('Otariidae', 3, [
                ('Callorhinus',1,[]),
                ('Eumetopias',1,[]),
                ('Zalophus',1,[])
            ]),
            ('Canidae', 3, [
                ('Vulpes',2,[]),
                ('Canis',1,[])
            ]),
            ('Hyaenidae',1,[
                ('Hyaena',1,[])
            ]),
            ('Herpestidae',1,[
                ('Suricata',1,[])
            ])
        ]),
        ('Artiodactyla', 25, [
            ('Bovidae',6,[
                ('Bos',3,[]),
                ('Bubalus',1,[]),
                ('Capra',1,[]),
                ('Ovis',1,[])
            ]),
            ('Camelidae',4,[
                ('Camelus',3,[]),
                ('Vicugna',1,[])
            ]),
            ('Delphinidae',4,[
                ('Globicephala',1,[]),
                ('Lagenorhynchus',1,[]),
                ('Orcinus',1,[]),
                ('Tursiops',1,[])
            ]),
            ('Cervidae',3,[
                ('Cervus',2,[]),
                ('Odocoileus',1,[])
            ]),
            ('Monodontidae',2,[
                ('Deplhinapterus',1,[]),
                ('Mondon',1,[])
            ]),
            ('Suidae',2,[
                ('Sus',1,[]),
                ('Phacochoerus',1,[])
            ]),
            ('Balaenopteridae',1,[
                ('Balaenoptera',1,[])
            ]),
            ('Lipotidae',1,[
                ('Lipotes',1,[])
            ]),
            ('Phocoenidae',1,[
                ('Phocoena',1,[])
            ]),
            ('Physeteridae',1,[
                ('Physene',1,[])
            ])
        ]),
        ('Rodentia', 24, [
            ('Muridae',8,[
                ('Mus',3,[]),
                ('Rattus',2,[]),
                ('Grammomys',1,[]),
                ('Mastomys',1,[]),
                ('Meriones',1,[])
            ]),
            ('Cricetidae',5,[
                ('Cricetulus',1,[]),
                ('Mycrotus',1,[]),
                ('Peromyscus',1,[]),
                ('Myodes',1,[]),
                ('Mesocricetus',1,[])
            ]),
            ('Sciuridae',3,[
                ('Ictidomys',1,[]),
                ('Marmota',1,[]),
                ('Urocitellus',1,[])
            ]),
            ('Bathyergidae',2,[
                ('Fukomys',1,[]),
                ('Heterocephalus',1,[])
            ]),
            ('Castoridae',1,[
                ('Castor',1,[])
            ]),
            ('Caviidae',1,[
                ('Cavia',1,[])
            ]),
            ('Chinchillidae',1,[
                ('Chinchilla',1,[])
            ]),
            ('Dipodidae',1,[
                ('Jaculus',1,[])
            ]),
            ('Spalacidae',1,[
                ('Nannospalax',1,[])
            ]),
            ('Octodontidae',1,[
                ('Octodon',1,[])
            ])
        ]),
        ('Primates', 23, [
            ('Cercopithecidae',11,[
                ('Macaca',3,[]),
                ('Rhinopithecus',2,[]),
                ('Cercocebus',1,[]),
                ('Chlorocebus',1,[]),
                ('Mandrillus',1,[]),
                ('Papio',1,[]),
                ('Theropithecus',1,[]),
                ('Trachypithecus',1,[])
            ]),
            ('Hominidae',4,[
                ('Pan',2,[]),
                ('Gorilla',1,[]),
                ('Pongo',1,[])
            ]),
            ('Cebidae',2,[
                ('Callithrix',1,[]),
                ('Sapajus',1,[])
            ]),
            ('Aotidae',1,[
                ('Aouts',1,[])
            ]),
            ('Tarsidae',1,[
                ('Carlito',1,[])
            ]),
            ('Cheirogaleidae',1,[
                ('Microcebus',1,[])
            ]),
            ('Hylobatidae',1,[
                ('Nomascus',1,[])
            ]),
            ('Indriidae',1,[
                ('Propithecus',1,[])
            ]),
            ('Lemuridae',1,[
                ('Lemur',1,[])
            ])
        ]),
        ('Chirpotera', 11, [
            ('Vespertilionidae',4,[
                ('Myotis',4,[])
            ]),
            ('Pteropodidae',3,[
                ('Pteropus',2,[]),
                ('Rousettus',1,[])
            ]),
            ('Phyllostomidae',2,[
                ('Artibues',1,[]),
                ('Desmodus',1,[])
            ]),
            ('Hipposideridae',1,[
                ('Hipposideros',1,[])
            ]),
            ('Rhinolophidae',1,[
                ('Rhinolopus',1,[])
            ])
        ])
    ]),
]

def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    fig=plt.figure(figsize=(12,12))
    ax = ax or plt.subplot(111, projection='polar')

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2],color=None)
        ax.text(0, 0, label, ha='center', va='center',fontsize=16)
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
                       edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            if label in ['Carnivora','Artiodactyla','Rodentia','Primates','Chiroptera']:
                ax.text(x, y, label, rotation=rotation, ha='center', va='center',fontsize=10)
            else:
                ax.text(x, y, label, rotation=rotation, ha='center', va='center',fontsize=8) 

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()
    return fig



fig=sunburst(data)

plt.tight_layout()
fig.savefig('../results/taxonomical_structure.png',dpi=800)
