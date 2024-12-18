export async function GET() {
  const response = await fetch('http://localhost:8000/neural_network_data')
  const data = await response.json()
  return Response.json(data)
}

