from Bio.Seq import SeqIO
import os
path = '/home/friederike/Documents/Coding/TailEnzA_main/TailEnzA/Classifiers/Training_data/Dataset/'
BGC_types=["NRPS","PKS"]
enzymes=[["ycao","ycaO","YCAO"],["P450","p450"],["radical SAM"],["methyltransferase"]]

proteins=[]
for BGC_type in BGC_types:
    for enzyme in enzymes:
        final_path=path+BGC_type+"_genbank_files_antismash_DB/"
        output_path=open(path+BGC_type+"_antismash_DB_"+enzyme[0]+".fasta","w")
        for progenome_file in os.listdir(final_path):
                try:
                    #for each genome
                    for seq_record in SeqIO.parse(final_path+progenome_file, "gb"):
                    
                        for name in enzyme:
                            for seq_feature in seq_record.features :
                           
                                if  seq_feature.type=="CDS" and name.lower() in str(seq_feature).lower():
                                  
                                    assert len(seq_feature.qualifiers['translation'])==1
                                    output_path.write(">%s from %s (%s)\n%s\n" % (
                                           seq_feature.qualifiers['locus_tag'][0],
                                           seq_record.name,
                                           seq_feature.qualifiers["product"],
                                           seq_feature.qualifiers['translation'][0]))
                                    print (seq_feature.qualifiers["product"])
                except: print ("error")
        output_path.close()