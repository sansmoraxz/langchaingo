package bedrockclient

import (
	"context"
	"encoding/json"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
)

// Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html


type cohereTextGenerationInput struct {
	Prompt string `json:"prompt"`
	Temperature float64 `json:"temperature"`
	P float64 `json:"p"`
	K int `json:"k"`
	MaxTokens int `json:"max_tokens"`
	StopSequences []string `json:"stop_sequences"`
	NumGenerations int `json:"num_generations"`
}


type cohereTextGenerationOutput struct {
	Generations []*cohereTextGenerationOutputGeneration `json:"generations"`
	ID string `json:"id"`
	Prompt string `json:"prompt"`
}

type cohereTextGenerationOutputGeneration struct {
	ID string `json:"id"`
	Index int `json:"index"`
	FinishReason string `json:"finish_reason"`
	Text string `json:"text"`
}

func createCohereCompletion(ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	txt := processInputMessagesGeneric(messages)

	input := &cohereTextGenerationInput{
		Prompt: txt,
		Temperature: options.Temperature,
		P: options.TopP,
		K: options.TopK,
		MaxTokens: options.MaxTokens,
		StopSequences: options.StopWords,
		NumGenerations: options.CandidateCount,
	}

	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	modelInput := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        body,
	}
	resp, err := client.InvokeModel(ctx, modelInput)

	if err != nil {
		return nil, err
	}

	var output cohereTextGenerationOutput

	err = json.Unmarshal(resp.Body, &output)
	if err != nil {
		return nil, err
	}

	choices := make([]*llms.ContentChoice, len(output.Generations))

	for i, gen := range output.Generations {
		choices[i] = &llms.ContentChoice{
			Content: gen.Text,
			StopReason: gen.FinishReason,
			GenerationInfo: map[string]interface{}{
				"generation_id": gen.ID,
				"index": i,
			},
		}
	}

	return &llms.ContentResponse{
		Choices: choices,
	}, nil
}
