package bedrock_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/bedrock"
	"github.com/tmc/langchaingo/schema"
)

var msgs []llms.MessageContent = []llms.MessageContent{
	{
		Role: schema.ChatMessageTypeSystem,
		Parts: []llms.ContentPart{
			llms.TextPart("You are a chatbot."),
		},
	},
	{
		Role: schema.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextPart("Explain AI in 10 words or less."),
		},
	},
}


// All the test models
var models = []string{
	bedrock.ModelAi21J2MidV1,
	bedrock.ModelAi21J2UltraV1,
	bedrock.ModelAmazonTitanTextLiteV1,
	bedrock.ModelAmazonTitanTextExpressV1,
	bedrock.ModelAnthropicClaude3Sonnet20240229V10,
	bedrock.ModelAnthropicClaudeV21,
	bedrock.ModelAnthropicClaudeV2,
	bedrock.ModelAnthropicClaudeInstantV1,
	bedrock.ModelCohereCommandTextV14,
	bedrock.ModelCohereCommandLightTextV14,
	bedrock.ModelMetaLlama213bChatV1,
	bedrock.ModelMetaLlama270bChatV1,
}

func setUpTest() (*bedrockruntime.Client, error) {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		return nil, err
	}
	client := bedrockruntime.NewFromConfig(cfg)
	return client, nil
}

func TestAmazonOutput(t *testing.T) {
	client, err := setUpTest()
	if err != nil {
		t.Fatal(err)
	}
	llm, err := bedrock.New(bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()

	for _, model := range models {

		fmt.Println("--------------------------------------------------")
		fmt.Printf("Model: %s\n", model)

		resp, err := llm.GenerateContent(ctx, msgs, llms.WithModel(model))
		if err != nil {
			t.Fatal(err)
		}
		for i, choice := range resp.Choices {
			fmt.Printf("Choice %d:\n", i)
			fmt.Printf("  Text: %s\n", choice.Content)
		}

		fmt.Println("--------------------------------------------------")
	}
}
