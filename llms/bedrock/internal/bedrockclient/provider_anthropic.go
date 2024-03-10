package bedrockclient

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
)

// Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

// Finish reason for the completion of the generation
const (
	AnthropicCompletionReasonEndTurn = "end_turn"
	AnthropicCompletionReasonMaxTokens = "max_tokens"
	AnthropicCompletionReasonStopSequence = "stop_sequence"
)


// The latest version of the model
const (
	AnthropicLatestVersion = "bedrock-2023-05-31"
)

// Role attribute for the anthropic message
const (
	AnthropicRoleUser = "user"
	AnthropicRoleAssistant = "assistant"
)

// Type attribute for the anthropic message
const (
	AnthropicMessageTypeText = "text"
	AnthropicMessageTypeImage = "image"
)

type anthropicTextGenerationInputSource struct {
	Type string `json:"type"`
	MediaType string `json:"media_type"`
	Data string `json:"data"`
}

type anthropicTextGenerationInputContent struct {
	Type string `json:"type"`
	Source *anthropicTextGenerationInputSource `json:"source,omitempty"`
	Text *string `json:"text,omitempty"`
}

type anthropicTextGenerationInputMessage struct {
	Role string `json:"role"`
	Content []anthropicTextGenerationInputContent `json:"content"`
}

type anthropicTextGenerationInput struct {
	AnthropicVersion string `json:"anthropic_version"`
	MaxTokens int `json:"max_tokens"`
	// older versions may not have this field
	System *string `json:"system,omitempty"`
	Messages []*anthropicTextGenerationInputMessage `json:"messages"`
	Temperature float32 `json:"temperature"`
	TopP float32 `json:"top_p"`
	TopK int `json:"top_k"`
	StopSequences []string `json:"stop_sequences"`
}


type anthropicTextGenerationOutput struct {
	Type string `json:"type"`
	Role string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	StopSequence string `json:"stop_sequence"`
	Usage struct {
		InputTokens int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func createAnthropicCompletion(ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	inputContent anthropicTextGenerationInput,
) (*llms.ContentResponse, error) {
	body, err := json.Marshal(inputContent)
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

	var output anthropicTextGenerationOutput
	err = json.Unmarshal(resp.Body, &output)
	if err != nil {
		return nil, err
	}

	if len(output.Content) == 0 {
		return nil, errors.New("no results")
	} else if stopReason := output.StopReason; stopReason != AnthropicCompletionReasonEndTurn && stopReason != AnthropicCompletionReasonStopSequence {
		return nil, errors.New("completed due to " + stopReason + ". Maybe try increasing max tokens")
	}
	Contentchoices := make([]*llms.ContentChoice, len(output.Content))
	for i, c := range output.Content {
		Contentchoices[i] = &llms.ContentChoice{
			Content: c.Text,
			StopReason: output.StopReason,
			GenerationInfo: map[string]interface{}{
				"input_tokens": output.Usage.InputTokens,
				"output_tokens": output.Usage.OutputTokens,
			},
		}
	}
	return &llms.ContentResponse{
		Choices: Contentchoices,
	}, nil
}
