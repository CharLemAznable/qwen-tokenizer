package tokenizer_test

import (
	"github.com/CharLemAznable/qwen-tokenizer"
	"github.com/gogf/gf/v2/test/gtest"
	"reflect"
	"testing"
)

func Test_EncodeOrdinary(t *testing.T) {
	gtest.C(t, func(t *gtest.T) {
		tz := &tokenizer.Tokenizer{}
		prompt := "如果现在要你走十万八千里路，需要多长的时间才能到达？ "
		ids := tz.EncodeOrdinary(prompt)
		expect := []int{62244, 99601, 30534, 56568, 99314, 110860, 99568, 107903, 45995, 3837, 85106, 42140, 45861, 101975, 101901, 104658, 11319, 220}
		t.Assert(reflect.DeepEqual(ids, expect), true)
		decodedString := tz.Decode(ids)
		t.Assert(decodedString, prompt)
	})
}

func Test_Encode(t *testing.T) {
	gtest.C(t, func(t *gtest.T) {
		tz := &tokenizer.Tokenizer{}
		prompt := "<|im_start|>system\nYour are a helpful assistant.<|im_end|>\n<|im_start|>user\nSanFrancisco is a<|im_end|>\n<|im_start|>assistant\n"
		ids := tz.Encode(prompt, tokenizer.All)
		expect := []int{151644, 8948, 198, 7771, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 23729, 80328, 9464, 374, 264, 151645, 198, 151644, 77091, 198}
		t.Assert(reflect.DeepEqual(ids, expect), true)
		decodedString := tz.Decode(ids)
		t.Assert(decodedString, prompt)
	})
}
