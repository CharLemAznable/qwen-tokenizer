package tokenizer

import (
	"bufio"
	_ "github.com/CharLemAznable/qwen-tokenizer/tiktoken"
	"github.com/dlclark/regexp2"
	"github.com/gogf/gf/v2/container/garray"
	"github.com/gogf/gf/v2/encoding/gbase64"
	"github.com/gogf/gf/v2/os/gres"
	"github.com/gogf/gf/v2/text/gstr"
	"github.com/gogf/gf/v2/util/gconv"
	"math"
	"strings"

	"fmt"
	"github.com/gogf/gf/v2/container/gmap"
)

const (
	SpecialStart       = "<|"
	SpecialEnd         = "|>"
	EndOfText          = "<|endoftext|>"
	ImStart            = "<|im_start|>"
	ImEnd              = "<|im_end|>"
	PattenString       = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	SpecialStartId     = 151643
	TokenRankSeparator = " "
	VocabularyBpeFile  = "resources/qwen.tiktoken"
)

var (
	mergeableRanks *gmap.ListMap
	specialTokens  *gmap.ListMap
	decodeMap      []string
)

func init() {
	specialTokens = gmap.NewListMap()
	specialStartIndex := SpecialStartId
	specialTokens.Set(EndOfText, specialStartIndex)
	specialStartIndex += 1
	specialTokens.Set(ImStart, specialStartIndex)
	specialStartIndex += 1
	specialTokens.Set(ImEnd, specialStartIndex)
	specialStartIndex += 1
	for i := 0; i < 205; i++ {
		specialToken := fmt.Sprintf("<|extra_%d|>", i)
		specialTokens.Set(specialToken, specialStartIndex)
		specialStartIndex += 1
	}

	// ref: https://github.com/openai/tiktoken/blob/main/tiktoken/load.py#L143
	mergeableRanks = gmap.NewListMap()
	scanner := bufio.NewScanner(gres.Get(VocabularyBpeFile))
	for scanner.Scan() {
		line := scanner.Text()
		splits := gstr.Split(line, TokenRankSeparator)
		if len(splits) != 2 {
			panic("Invalid line in " + VocabularyBpeFile + ": " + line)
		}
		token := string(gbase64.MustDecodeString(splits[0]))
		rank := gconv.Int(splits[1])
		mergeableRanks.Set(token, rank)
	}

	decodeMap = make([]string, mergeableRanks.Size()+specialTokens.Size())
	mergeableRanks.Iterator(func(key, value interface{}) bool {
		decodeMap[gconv.Int(value)] = gconv.String(key)
		return true
	})
	specialTokens.Iterator(func(key, value interface{}) bool {
		decodeMap[gconv.Int(value)] = gconv.String(key)
		return true
	})
}

type Tokenizer struct {
}

func (t *Tokenizer) EncodeOrdinary(text string) (tokens []int) {
	tokens = make([]int, 0)
	regexp := regexp2.MustCompile(PattenString, 0)
	match, err := regexp.FindStringMatch(text)
	for err == nil && match != nil {
		tokens = append(tokens, t.encodeChunk(match.String())...)
		match, err = regexp.FindNextMatch(match)
	}
	return
}

type SpecialAllowed string

const (
	All       SpecialAllowed = "all"
	None      SpecialAllowed = "none"
	NoneRaise SpecialAllowed = "none_raise"
)

func (t *Tokenizer) Encode(text string, allowedSpecial ...SpecialAllowed) (tokens []int) {
	special := All
	if len(allowedSpecial) > 0 && allowedSpecial[0] != "" {
		special = allowedSpecial[0]
	}
	var specialTokensUse *gmap.ListMap
	switch special {
	case All:
		specialTokensUse = specialTokens
	case None:
		specialTokensUse = gmap.NewListMap()
	case NoneRaise:
		specialTokensUse = gmap.NewListMap()
		isSpecialTokenExists := false
		for _, token := range specialTokens.Keys() {
			if gstr.Contains(text, gconv.String(token)) {
				isSpecialTokenExists = true
				break
			}
		}
		if !isSpecialTokenExists {
			panic("No special token in " + text)
		}
	default:
		panic("UnSupport allowedSpecial: " + special)
	}
	if specialTokensUse.IsEmpty() {
		return t.EncodeOrdinary(text)
	}
	chunks := t.splitWithSpecial(text)
	tokens = make([]int, 0)
	for _, chunk := range chunks {
		if specialTokensUse.Contains(chunk) {
			tokens = append(tokens, gconv.Int(specialTokensUse.Get(chunk)))
		} else {
			tokens = append(tokens, t.EncodeOrdinary(chunk)...)
		}
	}
	return
}

func (t *Tokenizer) Decode(tokens []int) string {
	var buffer strings.Builder
	for _, token := range tokens {
		buffer.WriteString(decodeMap[token])
	}
	return buffer.String()
}

func (t *Tokenizer) encodeChunk(chunk string) (tokens []int) {
	chunkBytes := []byte(chunk)
	ids := make([]*entity, len(chunkBytes))
	for idx, b := range chunkBytes {
		e := newEntity([]byte{b})
		e.rank = gconv.Int(mergeableRanks.Get(string(b)))
		ids[idx] = e
	}
	if len(ids) < 2 {
		for _, e := range ids {
			tokens = append(tokens, e.rank)
		}
		return
	}
	for len(ids) >= 2 {
		bytePair := t.getLowestIndexBytePair(ids)
		if bytePair == nil {
			break
		}
		ids = t.merge(ids, bytePair)
	}
	for _, e := range ids {
		tokens = append(tokens, e.rank)
	}
	return
}

func (t *Tokenizer) getLowestIndexBytePair(ids []*entity) (minRankPair *entity) {
	var (
		bytePairs = garray.NewStrArray()
		minRank   = math.MaxInt32
	)
	for i := 0; i < len(ids)-1; i++ {
		bytePair := t.mergePair(ids[i], ids[i+1])
		token := string(bytePair.bytes)
		if bytePairs.Contains(token) {
			continue
		}
		rank := mergeableRanks.Get(token)
		if rank != nil {
			if r := gconv.Int(rank); r < minRank {
				minRank = r
				minRankPair = &entity{
					bytes: []byte(token),
					rank:  r,
				}
			}
		}
		bytePairs.Append(token)
	}
	return
}

func (t *Tokenizer) merge(ids []*entity, bytePair *entity) []*entity {
	merged := make([]*entity, len(ids))
	mergedIndex := 0
	for i := 0; i < len(ids); {
		if i < len(ids)-1 {
			mergePair := t.mergePair(ids[i], ids[i+1])
			if string(mergePair.bytes) == string(bytePair.bytes) {
				merged[mergedIndex] = bytePair
				mergedIndex += 1
				i += 2
			} else {
				merged[mergedIndex] = ids[i]
				mergedIndex += 1
				i += 1
			}
		} else {
			merged[mergedIndex] = ids[i]
			mergedIndex += 1
			i += 1
		}
	}
	return merged[0:mergedIndex]
}

func (t *Tokenizer) mergePair(first, second *entity) *entity {
	return newEntity(append(first.bytes, second.bytes...))
}

func (t *Tokenizer) splitWithSpecial(text string) []string {
	if gstr.Contains(text, SpecialStart) && gstr.Contains(text, SpecialEnd) {
		return splitByStrings(text, specialTokens.Keys())
	} else {
		return []string{text}
	}
}

func splitByStrings(text string, spliters []interface{}) (chunks []string) {
	chunks = []string{text}
	for _, spliter := range spliters {
		thisSplits := make([]string, 0)
		for _, chunk := range chunks {
			thisSplits = append(thisSplits, splitByString(chunk, gconv.String(spliter))...)
		}
		chunks = thisSplits
	}
	return
}

func splitByString(src, spliter string) (parts []string) {
	parts = make([]string, 0)
	from := 0
	first := gstr.Pos(src, spliter, from)
	for first != -1 {
		if from == first {
			parts = append(parts, spliter)
			from += len(spliter)
		} else {
			parts = append(parts, src[from:first], spliter)
			from += first - from + len(spliter)
		}
		first = gstr.Pos(src, spliter, from)
	}
	remain := src[from:]
	if remain != "" {
		parts = append(parts, remain)
	}
	return
}

type entity struct {
	bytes []byte
	rank  int
}

func newEntity(bytes []byte) *entity {
	return &entity{
		bytes: bytes,
		rank:  math.MaxInt32,
	}
}
