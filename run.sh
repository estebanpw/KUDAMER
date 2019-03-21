#!/bin/bash

QUERY=$1
REF=$2
DEV=$3


if [ $# != 3 ]; then
	echo "***ERROR*** Use: $0 query ref device"
	exit -1
fi

filename=$(basename -- "$QUERY")
extensionA="${filename##*.}"

filename=$(basename -- "$REF")
extensionB="${filename##*.}"

indexA=$(basename "$QUERY")
indexB=$(basename "$REF")

BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PATHX=$(dirname "${QUERY}")
PATHY=$(dirname "${REF}")

SEQNAMEX="${indexA%.*}"
SEQNAMEY="${indexB%.*}"

if [ ! -f ${PATHX}/${SEQNAMEX}.${extensionA}.fix ]; then

	$BINDIR/pre-process.sh $QUERY
fi

if [ ! -f ${PATHY}/${SEQNAMEY}.${extensionB}.fix ]; then

	$BINDIR/pre-process.sh $REF
fi

$BINDIR/index_kmers_split_dyn_mat -query ${PATHX}/${SEQNAMEX}.${extensionA}.fix -ref ${PATHY}/${SEQNAMEY}.${extensionB}.fix -dev $DEV

#echo "Plotting algorithm currently disabled"
#echo "(Rscript --vanilla $BINDIR/plot_and_score.R ${SEQNAMEX}.${extensionA}-${SEQNAMEY}.${extensionB}.mat $DIM) &> ${SEQNAMEX}.${extensionA}-${SEQNAMEY}.${extensionB}.scr.txt"

# (Rscript $BINDIR/compute_score.R $FILE1-$FILE2.mat $DIM) &> $FILE1-$FILE2.scr.txt

#rm ${PATHX}/${SEQNAMEX}.fix.fasta
#rm ${PATHY}/${SEQNAMEY}.fix.fasta


