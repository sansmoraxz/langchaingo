package bedrockclient

import (
	"context"
	"encoding/json"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
)

type ai21TextGenerationInput struct {
	Prompt string `json:"prompt"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP float64 `json:"topP,omitempty"`
	MaxTokens int `json:"maxTokens,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
	CountPenalty struct {
		Scale float64 `json:"scale"`
	} `json:"countPenalty"`
	PresencePenalty struct {
		Scale float64 `json:"scale"`
	} `json:"presencePenalty"`
	FrequencyPenalty struct {
		Scale float64 `json:"scale"`
	} `json:"frequencyPenalty"`
}


type ai21TextGenerationOutput struct {
	ID any `json:"id"` // Docs say it's a string, got number
	Prompt struct {
		Text string `json:"text"`
		Tokens []struct{} `json:"tokens"` // for counting only
	} `json:"prompt"`
	Completions []struct {
		Data struct {
			Text string `json:"text"`
			Tokens []struct{} `json:"tokens"` // for counting only
		} `json:"data"`
		FinishReason struct {
			Reason string `json:"reason"`
		} `json:"finishReason"`
	} `json:"completions"`
}

func createAi21Completion(ctx context.Context, client *bedrockruntime.Client, modelID string, messages []Message, options llms.CallOptions) (*llms.ContentResponse, error) {
	txt := processInputMessagesGeneric(messages)
	inputContent := ai21TextGenerationInput{
		Prompt: txt,
		Temperature: options.Temperature,
		TopP: options.TopP,
		MaxTokens: options.MaxTokens,
		StopSequences: options.StopWords,
		CountPenalty: struct {
			Scale float64 `json:"scale"`
		}{Scale: options.RepetitionPenalty},
		PresencePenalty: struct {
			Scale float64 `json:"scale"`
		}{Scale: 0},
		FrequencyPenalty: struct {
			Scale float64 `json:"scale"`
		}{Scale: 0},
	}

	body, err := json.Marshal(inputContent)
	if err != nil {
		return nil, err
	}

	modelInput := bedrockruntime.InvokeModelInput{
		ModelId: aws.String(modelID),
		Body:    body,
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
	}

	resp, err := client.InvokeModel(ctx, &modelInput)
	if err != nil {
		return nil, err
	}

	var output ai21TextGenerationOutput
	err = json.Unmarshal(resp.Body, &output)
	if err != nil {
		return nil, err
	}

	choices := make([]*llms.ContentChoice, len(output.Completions))
	for i, completion := range output.Completions {
		choices[i] = &llms.ContentChoice{
			Content: completion.Data.Text,
			StopReason: completion.FinishReason.Reason,
			GenerationInfo: map[string]any{
				"id": output.ID,
				"input_tokens": len(output.Prompt.Tokens),
				"output_tokens": len(completion.Data.Tokens),
			},
		}
	}

	return &llms.ContentResponse{Choices: choices}, nil
}
