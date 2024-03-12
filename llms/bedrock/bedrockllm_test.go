package bedrock_test

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/bedrock"
	"github.com/tmc/langchaingo/schema"
)

func setUpTest() (*bedrockruntime.Client, error) {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		return nil, err
	}
	client := bedrockruntime.NewFromConfig(cfg)
	return client, nil
}

func TestAmazonOutput(t *testing.T) {
	t.Parallel()

	client, err := setUpTest()
	if err != nil {
		t.Fatal(err)
	}
	llm, err := bedrock.New(bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}

	msgs := []llms.MessageContent{
		{
			Role: schema.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{
				llms.TextPart("You know all about AI."),
			},
		},
		{
			Role: schema.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Explain AI in 10 words or less."),
			},
		},
	}

	// All the test models.
	models := []string{
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

	ctx := context.Background()

	for _, model := range models {
		t.Log("\n--------------------------------------------------\n")
		t.Logf("Model: %s\n", model)

		resp, err := llm.GenerateContent(ctx, msgs, llms.WithModel(model), llms.WithMaxTokens(512))
		if err != nil {
			t.Fatal(err)
		}
		for i, choice := range resp.Choices {
			t.Logf("Choice %d:-\n", i)
			t.Logf("%s\n", choice.Content)
		}

		t.Log("\n--------------------------------------------------\n")
	}
}
