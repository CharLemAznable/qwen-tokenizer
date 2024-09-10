package tokenizer

import (
	"context"
	"github.com/gogf/gf/v2/frame/g"
	"github.com/gogf/gf/v2/os/gmutex"
)

const (
	loggerName = "qwen_token"
)

type TokenCounter struct {
	tokenizer *Tokenizer
	deviation *deviation
}

func NewTokenCounter(name string, initMean ...float64) *TokenCounter {
	return &TokenCounter{
		tokenizer: &Tokenizer{},
		deviation: newDeviation(name, initMean...),
	}
}

func (c *TokenCounter) Count(text string) int {
	return len(c.tokenizer.Encode(text))
}

func (c *TokenCounter) Correct(count int) int {
	return int(c.deviation.Get() * float64(count))
}

func (c *TokenCounter) Update(actualCount, count int) {
	c.deviation.Update(float64(actualCount) / float64(count))
}

type deviation struct {
	name  string
	mutex *gmutex.RWMutex
	count uint64
	mean  float64
}

func newDeviation(name string, initMean ...float64) *deviation {
	count, mean := uint64(0), float64(0)
	if len(initMean) > 0 {
		count, mean = 1, initMean[0]
	}
	return &deviation{
		name:  name,
		mutex: &gmutex.RWMutex{},
		count: count,
		mean:  mean,
	}
}

func (d *deviation) Get() (m float64) {
	d.mutex.RLockFunc(func() {
		m = d.mean
	})
	return
}

func (d *deviation) Update(value float64) {
	d.mutex.LockFunc(func() {
		d.count++
		d.mean = d.mean + (value-d.mean)/float64(d.count)
		g.Log(loggerName).Async().Debugf(context.Background(),
			"deviation [%s] mean: [%f], update with [%f]", d.name, d.mean, value)
	})
}
