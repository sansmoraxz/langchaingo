package bedrock

import (
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/callbacks"
)


type options struct {
	modelId string
	client *bedrockruntime.Client
	callbackHandler callbacks.Handler
}


// WithModel allows setting a custom modelId.
func WithModel(modelId string) Option {
	return func(o *options) {
		o.modelId = modelId
	}
}


// WithCallback allows setting a custom Callback Handler.
func WithCallback(callbackHandler callbacks.Handler) Option {
	return func(o *options) {
		o.callbackHandler = callbackHandler
	}
}


type Option func(*options)