from functionalities import Compare

MtSizes=pd.read_csv("MtSizes.csv",index_col=0)["mt_size"]
org1,org2="Rattus_norvegicus","Homo_sapiens"
orgs=[org2,org1]

connection=sqlite3.connect('MANUDBrev.db')
compare=Compare(connection=connection)
Compdf=compare.get_compdf(MtSizes=MtSizes,orgs=orgs)
Identitydf=compare.get_seq_identity(orgs=orgs)

fig=plt.figure(figsize=(8,10))

# First row with 4 square subplots
ax1=plt.subplot2grid(shape=(3,4), loc=(0, 0), colspan=1)  # (row, col)
compare.boxplot(Compdf=Compdf,orgs=orgs,y_name="NUMT size (bp)",ax=ax1)
ax1.set_title('A',fontsize=20,x=-.25,y=1.05)

ax2=plt.subplot2grid(shape=(3,4), loc=(0, 1), colspan=1)
compare.boxplot(Compdf=Identitydf,orgs=orgs,y_name="Sequence identity",ax=ax2)
ax2.set_title('B',fontsize=20,x=-.25,y=1.05)

Regdf1,Regdf2=compare.get_regdf(Compdf=Compdf,orgs=orgs)
ax3=plt.subplot2grid(shape=(3,4), loc=(0, 2), colspan=1)
compare.regplot(Regdf=Regdf1,color="lightblue",ax=ax3)
ax3.set_title('C',fontsize=20,x=-.25,y=1.05)

ax4=plt.subplot2grid(shape=(3,4), loc=(0, 3), colspan=1)
compare.regplot(Regdf=Regdf2,color="orange",ax=ax4)
ax4.set_title('D',fontsize=20,x=-.25,y=1.05)

ax5=plt.subplot2grid(shape=(3,4), loc=(1, 0), colspan=2)
compare.histplot(Compdf=Compdf,org=orgs[0],color="lightblue",MtSizes=MtSizes,ax=ax5)
ax5.set_title('E',fontsize=20,x=-.25,y=1.05)

ax6=plt.subplot2grid(shape=(3,4), loc=(1, 2), colspan=2)
compare.histplot(Compdf=Compdf,org=orgs[1],color="orange",MtSizes=MtSizes,ax=ax6)
ax6.set_title('F',fontsize=20,x=-.25,y=1.05)

handles = [
    plt.Line2D([0], [0], color="lightblue", lw=4, label=f"""{orgs[0]} ({orgs[0][:2]} {orgs[0].split("_")[1][:2]})"""),
    plt.Line2D([0], [0], color="orange", lw=4, label=f"""{orgs[1]} ({orgs[1][:2]} {orgs[1].split("_")[1][:2]})""")
]
fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig("SampleComparative.png",dpi=400)