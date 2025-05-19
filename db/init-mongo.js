db = db.getSiblingDB('voice_auth');

db.createUser({
  user: 'mongo',
  pwd: 'password123',
  roles: [
    {
      role: 'readWrite',
      db: 'voice_auth'
    }
  ]
});