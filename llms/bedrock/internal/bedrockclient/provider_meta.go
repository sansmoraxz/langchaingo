package bedrockclient

import (
	"context"
	"encoding/json"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
)

type metaTextGenerationInput struct {
	Prompt string  `json:"prompt"`
	Temperature float64 `json:"temperature"`
	TopP float64 `json:"top_p"`
	MaxGenLen int `json:"max_gen_len"`
}

type metaTextGenerationOutput struct {
	Generation string `json:"generation"`
	PromptTokenCount int `json:"prompt_token_count"`
	GenerationTokenCount int `json:"generation_token_count"`
	StopReason string `json:"stop_reason"`
}

func createMetaCompletion(ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	txt := processInputMessagesGeneric(messages)

	input := &metaTextGenerationInput{
		Prompt: txt,
		Temperature: options.Temperature,
		TopP: options.TopP,
		MaxGenLen: options.MaxTokens,
	}

	body, err := json.Marshal(input)

	if err != nil {
		return nil, err
	}

	modelInput := &bedrockruntime.InvokeModelInput{
		ModelId: aws.String(modelID),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        body,
	}

	resp, err := client.InvokeModel(ctx, modelInput)
	if err != nil {
		return nil, err
	}

	var output metaTextGenerationOutput

	err = json.Unmarshal(resp.Body, &output)
	if err != nil {
		return nil, err
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content: output.Generation,
				StopReason: output.StopReason,
				GenerationInfo: map[string]interface{}{
					"input_tokens": output.PromptTokenCount,
					"output_tokens": output.GenerationTokenCount,
				},
			},
		},
	}, nil

}
