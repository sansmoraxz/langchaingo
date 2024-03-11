package bedrockclient

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)



const (
	AmazonCompletionReasonFinish = "FINISH"
	AmazonCompletionReasonMaxTokens = "LENGTH"
	AmazonCompletionReasonContentFiltered = "CONTENT_FILTERED"
)

// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html


type amazonTextGenerationConfigInput struct {
	MaxTokens int `json:"maxTokenCount"`
	TopP float64 `json:"topP"`
	Temperature float64 `json:"temperature"`
	StopSequences []string `json:"stopSequences"`
}

type amazonTextGenerationInput struct {
	InputText string `json:"inputText"`
	TextGenerationConfig amazonTextGenerationConfigInput `json:"textGenerationConfig"`
}


type amazonTextGenerationOutput struct {
	InputTextTokenCount int `json:"inputTextTokenCount"`
	Results []struct {
		TokenCount int `json:"tokenCount"`
		OutputText string `json:"outputText"`
		CompletionReason string `json:"completionReason"`
	} `json:"results"`
}

func createAmazonCompletion(ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	var messageContent strings.Builder
	for _, message := range messages {
		if message.Type != "text" {
			continue
		}
		if message.Role != "" {
			role, err := getTitanRole(message.Role)
			if err != nil {
				return nil, err
			}
			messageContent.WriteString(role + ": ")
		}
		messageContent.WriteString(message.Content)
		messageContent.WriteString("\n")
	}

	messageContent.WriteString("\nBot:")

	inputContent := amazonTextGenerationInput{
		InputText: messageContent.String(),
		TextGenerationConfig: amazonTextGenerationConfigInput{
			MaxTokens: options.MaxTokens,
			TopP: options.TopP,
			Temperature: options.Temperature,
			StopSequences: options.StopWords,
		},
	}

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

	var output amazonTextGenerationOutput
	err = json.Unmarshal(resp.Body, &output)
	if err != nil {
		return nil, err
	}

	if len(output.Results) == 0 {
		return nil, errors.New("no results")
	}

	contentChoices := make([]*llms.ContentChoice, len(output.Results))

	for i, result := range output.Results {
		contentChoices[i] = &llms.ContentChoice{
			Content: result.OutputText,
			StopReason: result.CompletionReason,
			GenerationInfo: map[string]any{
				"input_tokens": output.InputTextTokenCount,
				"output_tokens": result.TokenCount,
			},
		}
	}

	return &llms.ContentResponse{
		Choices: contentChoices,
	}, nil
}


func getTitanRole(role schema.ChatMessageType) (string, error) {
	switch role {
	case schema.ChatMessageTypeSystem:
		return "System", nil

	case schema.ChatMessageTypeAI:
		return "Bot", nil

	case schema.ChatMessageTypeGeneric:
		fallthrough
	case schema.ChatMessageTypeHuman:
		return "User", nil

	case schema.ChatMessageTypeFunction:
		fallthrough
	default:
		return "", errors.New("role not supported")
	}
}
