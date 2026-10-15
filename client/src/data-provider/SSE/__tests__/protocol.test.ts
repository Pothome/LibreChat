const mockAuthenticatedFetch = jest.fn();

jest.mock('librechat-data-provider', () => ({
  request: {
    authenticatedFetch: (...args: unknown[]) => mockAuthenticatedFetch(...args),
  },
}));

import { GENERATION_PROTOCOL_HEADER, postGenerationRequest } from '../protocol';

const response = (body: unknown, status = 200): Response =>
  ({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'Retry-After': '3' }),
    text: jest.fn().mockResolvedValue(body == null ? '' : JSON.stringify(body)),
  }) as unknown as Response;

describe('generation protocol negotiation', () => {
  beforeEach(() => {
    mockAuthenticatedFetch.mockReset();
  });

  it('advertises v2 in both the request header and JSON body', async () => {
    mockAuthenticatedFetch.mockResolvedValueOnce(
      response({ streamId: 'stream-1', generationProtocolVersion: 2 }),
    );

    await expect(postGenerationRequest('/api/agents/chat', { text: 'hello' })).resolves.toEqual({
      streamId: 'stream-1',
      generationProtocolVersion: 2,
    });

    expect(mockAuthenticatedFetch).toHaveBeenCalledWith(
      '/api/agents/chat',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          [GENERATION_PROTOCOL_HEADER]: '2',
        }),
        body: JSON.stringify({ text: 'hello', generationProtocolVersion: 2 }),
      }),
    );
  });

  it('keeps Axios-compatible status, body, and headers for existing error callers', async () => {
    mockAuthenticatedFetch.mockResolvedValueOnce(
      response({ code: 'SERVER_NOT_READY', message: 'warming up' }, 503),
    );

    await expect(postGenerationRequest('/api/agents/chat', {})).rejects.toMatchObject({
      message: 'warming up',
      response: {
        status: 503,
        data: { code: 'SERVER_NOT_READY', message: 'warming up' },
        headers: { 'retry-after': '3' },
      },
    });
  });
});
