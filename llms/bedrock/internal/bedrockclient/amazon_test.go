package bedrockclient

import (
	"context"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

func TestFetchAmazonOutput(t *testing.T) {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	client := bedrockruntime.NewFromConfig(cfg)

	c := NewClient(client)
	msg := []Message{
		{
			Role: "User",
			Content: "Who is the second president of the United States?",
			Type: "text",
		},
	}

	modelId := "meta.llama2-13b-chat-v1"

	opts := llms.CallOptions{
		CandidateCount: 3,
		// MaxTokens: 2000,
		// TopP: 1,
		// TopK: 0,
		// Temperature: 0.5,
		// CandidateCount: 1,
		// StopWords: []string{},
	}

	resp, err := c.CreateCompletion(context.Background(), modelId, msg, opts)
	if err != nil {
		t.Fatal(err)
	}
	
	for i, choice := range resp.Choices {
		fmt.Printf("\n\nOutput %d:\n%+v\n", i, choice.Content)
	}
}

func TestFetchAnthropicOutput(t *testing.T) {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	client := bedrockruntime.NewFromConfig(cfg)

	msg := Message{
		Role: schema.ChatMessageTypeHuman,
		Content: "Who is the second president of the United States?",
		Type: "text",
	}

	opts := llms.CallOptions{
		// MaxTokens: 2000,
		// TopP: 0.5,
		// Temperature: 0.5,
		// StopWords: []string{},
	}


	resp, err := createAnthropicCompletion(context.Background(), client, "anthropic.claude-instant-v1", []Message{msg}, opts)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("%+v\n", resp.Choices[0].Content)
}
